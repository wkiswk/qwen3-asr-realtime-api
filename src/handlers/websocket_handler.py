import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from models.asr_manager import ASRManager
from models.asr_session import ASRSession
from models.vad import VADManager
from utils.audio import bytes_to_numpy, decode_base64_audio
from utils.logger import get_logger
from utils.protocol import (
    create_conversation_item_created_event,
    create_error_event,
    create_input_audio_buffer_committed_event,
    create_session_created_event,
    create_session_finished_event,
    create_session_updated_event,
    create_speech_started_event,
    create_speech_stopped_event,
    create_transcription_completed_event,
    create_transcription_failed_event,
    create_transcription_text_event,
    generate_item_id,
    generate_session_id,
)

logger = get_logger(__name__)

STREAMING_CHUNK_SIZE_SEC = float(os.getenv("STREAMING_CHUNK_SIZE_SEC", "2.0"))
# Auto-commit interval in seconds (to prevent memory overflow on long continuous speech)
AUTO_COMMIT_INTERVAL_SEC = float(os.getenv("AUTO_COMMIT_INTERVAL_SEC", "60.0"))
# Default VAD silence duration in milliseconds
DEFAULT_VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "400"))

# Language code mapping: ISO code -> Qwen3-ASR language name
LANGUAGE_CODE_MAP = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "it": "Italian",
    "hi": "Hindi",
    "id": "Indonesian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "cs": "Czech",
    "da": "Danish",
    "fil": "Filipino",
    "fi": "Finnish",
    "is": "Icelandic",
    "ms": "Malay",
    "no": "Norwegian",
    "pl": "Polish",
    "sv": "Swedish",
    "nl": "Dutch",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ru": "Russian",
}


def normalize_language(language: Optional[str]) -> Optional[str]:
    """Convert ISO language code to Qwen3-ASR language name."""
    if not language:
        return None

    # If it's already a full name (starts with uppercase), return as-is
    if language[0].isupper():
        return language

    # If language is 'auto', return None to let the model auto-detect
    if language.lower() == "auto":
        return None

    # Map ISO code to full name
    return LANGUAGE_CODE_MAP.get(language.lower(), language)


class WebSocketHandler:
    def __init__(self, websocket: WebSocket, asr_manager: ASRManager):
        self.websocket = websocket
        self.asr_manager = asr_manager

        self.session_id: str = generate_session_id()
        self.model_name: str = "qwen3-asr-flash-realtime"

        self.vad_enabled: bool = True
        self.vad_threshold: float = 0.5
        self.vad_silence_ms: int = DEFAULT_VAD_SILENCE_DURATION_MS

        self.audio_format: str = "pcm"
        self.sample_rate: int = 16000
        self.language: Optional[str] = None

        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.total_samples: int = 0

        self.vad_manager: Optional[VADManager] = None
        self.asr_session: Optional[ASRSession] = None

        self.current_item_id: Optional[str] = None
        self.previous_item_id: str = ""
        self.speech_active: bool = False

        self.last_commit_time: float = 0.0
        self.segment_start_time: float = 0.0

        self.finished: bool = False

    async def handle(self):
        await self.websocket.accept()
        logger.info(f"WebSocket connection accepted: {self.session_id}")

        await self._send_event(
            create_session_created_event(
                session_id=self.session_id,
                model=self.model_name,
                input_audio_format=self.audio_format,
                turn_detection={
                    "type": "server_vad",
                    "threshold": self.vad_threshold,
                    "silence_duration_ms": self.vad_silence_ms,
                },
            )
        )

        try:
            while not self.finished:
                message = await self.websocket.receive_text()
                await self._handle_message(json.loads(message))
        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {self.session_id}")
        except Exception as e:
            logger.error(f"Error handling websocket: {e}")
            await self._send_error("internal_error", "server_error", str(e))
        finally:
            await self._cleanup()

    async def _handle_message(self, message: Dict[str, Any]):
        event_type = message.get("type", "")
        event_id = message.get("event_id", "unknown")

        if not isinstance(event_type, str):
            event_type = str(event_type)

        logger.debug(f"Received event: {event_type} (id: {event_id})")

        handlers = {
            "session.update": self._handle_session_update,
            "input_audio_buffer.append": self._handle_audio_append,
            "input_audio_buffer.commit": self._handle_audio_commit,
            "session.finish": self._handle_session_finish,
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            await self._send_error(
                "invalid_request_error",
                "invalid_event",
                f"Unknown event type: {event_type}",
                event_id=event_id,
            )

    async def _handle_session_update(self, message: Dict[str, Any]):
        session_data = message.get("session", {})

        if "input_audio_format" in session_data:
            self.audio_format = session_data["input_audio_format"]

        if "sample_rate" in session_data:
            self.sample_rate = session_data["sample_rate"]

        transcription_config = session_data.get("input_audio_transcription", {})
        if transcription_config:
            self.language = normalize_language(transcription_config.get("language"))

        turn_detection = session_data.get("turn_detection")
        if turn_detection is None:
            self.vad_enabled = False
        else:
            self.vad_enabled = True
            self.vad_threshold = turn_detection.get("threshold", 0.5)
            self.vad_silence_ms = turn_detection.get(
                "silence_duration_ms", DEFAULT_VAD_SILENCE_DURATION_MS
            )

        self.vad_manager = VADManager(
            enabled=self.vad_enabled,
            threshold=self.vad_threshold,
            silence_duration_ms=self.vad_silence_ms,
            sample_rate=self.sample_rate,
        )

        self.asr_session = ASRSession(
            asr_manager=self.asr_manager,
            language=self.language,
            sample_rate=self.sample_rate,
            chunk_size_sec=STREAMING_CHUNK_SIZE_SEC,
        )

        await self._send_event(
            create_session_updated_event(
                session_id=self.session_id,
                model=self.model_name,
                input_audio_format=self.audio_format,
                turn_detection={
                    "type": "server_vad",
                    "threshold": self.vad_threshold,
                    "silence_duration_ms": self.vad_silence_ms,
                }
                if self.vad_enabled
                else None,
            )
        )

    async def _handle_audio_append(self, message: Dict[str, Any]):
        audio_b64 = message.get("audio", "")
        if not audio_b64:
            return

        audio_bytes = decode_base64_audio(audio_b64)
        if audio_bytes is None:
            await self._send_error(
                "invalid_request_error", "invalid_audio", "Failed to decode base64 audio"
            )
            return

        audio_array = bytes_to_numpy(audio_bytes, self.audio_format, self.sample_rate)
        if audio_array is None:
            await self._send_error(
                "invalid_request_error",
                "invalid_audio_format",
                f"Failed to decode audio format: {self.audio_format}",
            )
            return

        self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
        self.total_samples += len(audio_array)

        current_time = time.time()

        if self.vad_enabled and self.vad_manager:
            await self._process_vad(audio_array)
        elif not self.vad_enabled and not self.current_item_id:
            self.current_item_id = generate_item_id()
            self.segment_start_time = current_time
            await self._send_event(
                create_conversation_item_created_event(
                    item_id=self.current_item_id, previous_item_id=self.previous_item_id
                )
            )

        if self.asr_session:
            await self.asr_session.append_audio(audio_array)
            interim_result = await self.asr_session.get_interim_result()
            if interim_result:
                await self._send_transcription_text(interim_result)

        # 在VAD模式下禁用自动提交，因为VAD会自动检测语音边界
        if not self.vad_enabled:
            await self._check_auto_commit(current_time)

    async def _process_vad(self, audio_chunk: np.ndarray):
        if self.vad_manager is None:
            return
        try:
            result = self.vad_manager.process(audio_chunk, self.total_samples)

            if result.get("speech_started") and not self.speech_active:
                self.speech_active = True
                self.current_item_id = generate_item_id()
                self.segment_start_time = time.time()
                await self._send_event(
                    create_speech_started_event(
                        audio_start_ms=result["audio_start_ms"], item_id=self.current_item_id
                    )
                )

            if result.get("speech_stopped"):
                if self.speech_active:
                    await self._handle_speech_stopped(result["audio_end_ms"])
                else:
                    await self._send_event(
                        create_speech_stopped_event(
                            audio_end_ms=result["audio_end_ms"], item_id=None
                        )
                    )
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            import traceback

            logger.error(traceback.format_exc())

    async def _handle_speech_stopped(self, audio_end_ms: int):
        if not self.speech_active or not self.current_item_id:
            return

        self.speech_active = False

        # 立即发送speech_stopped消息
        await self._send_event(
            create_speech_stopped_event(audio_end_ms=audio_end_ms, item_id=self.current_item_id)
        )

        # 然后执行_commit_audio进行ASR推理
        await self._commit_audio()

    async def _handle_audio_commit(self, message: Dict[str, Any]):
        if self.vad_enabled:
            await self._send_error(
                "invalid_request_error",
                "commit_not_allowed",
                "input_audio_buffer.commit is not allowed in VAD mode",
            )
            return

        if not self.current_item_id:
            self.current_item_id = generate_item_id()

        await self._commit_audio()

    async def _commit_audio(self):
        if not self.current_item_id:
            return

        await self._send_event(
            create_input_audio_buffer_committed_event(
                previous_item_id=self.previous_item_id, item_id=self.current_item_id
            )
        )

        await self._send_event(
            create_conversation_item_created_event(
                item_id=self.current_item_id, previous_item_id=self.previous_item_id
            )
        )

        if self.asr_session:
            final_result = await self.asr_session.finish()
            logger.info(
                f"ASR final result: {len(final_result['transcript'])}, language={final_result['language']}, emotion={final_result['emotion']}"
            )
            await self._send_transcription_completed(final_result)

        self.previous_item_id = self.current_item_id
        self.current_item_id = None

        if self.asr_session:
            await self.asr_session.reset()
        if self.vad_manager:
            self.vad_manager.reset()

        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_commit_time = time.time()
        self.segment_start_time = self.last_commit_time

    async def _check_auto_commit(self, current_time: float):
        if not self.current_item_id:
            return

        if self.segment_start_time == 0:
            self.segment_start_time = current_time
            return

        elapsed = current_time - self.segment_start_time
        if elapsed >= AUTO_COMMIT_INTERVAL_SEC:
            logger.info(f"Auto-commit triggered after {elapsed:.1f}s")
            await self._auto_commit_and_continue()

    async def _auto_commit_and_continue(self):
        if not self.current_item_id:
            return

        await self._send_event(
            create_input_audio_buffer_committed_event(
                previous_item_id=self.previous_item_id, item_id=self.current_item_id
            )
        )

        if self.asr_session:
            final_result = await self.asr_session.finish()
            logger.info(
                f"Auto-commit ASR result: {len(final_result['transcript'])}, language={final_result['language']}, emotion={final_result['emotion']}"
            )
            await self._send_transcription_completed(final_result)
            await self.asr_session.reset()

        self.previous_item_id = self.current_item_id
        self.current_item_id = generate_item_id()
        self.segment_start_time = time.time()
        self.last_commit_time = self.segment_start_time
        self.audio_buffer = np.array([], dtype=np.float32)

        await self._send_event(
            create_conversation_item_created_event(
                item_id=self.current_item_id, previous_item_id=self.previous_item_id
            )
        )

    async def _handle_session_finish(self, message: Dict[str, Any]):
        logger.info(f"Session finish requested: {self.session_id}")

        if self.vad_manager and self.speech_active:
            force_stop = self.vad_manager.force_stop(self.total_samples)
            if force_stop:
                await self._handle_speech_stopped(force_stop["audio_end_ms"])
        elif self.current_item_id:
            await self._commit_audio()

        await self._send_event(create_session_finished_event())
        self.finished = True

    async def _send_transcription_text(self, result: Dict[str, str]):
        if not self.current_item_id:
            return

        await self._send_event(
            create_transcription_text_event(
                item_id=self.current_item_id,
                content_index=0,
                language=result.get("language", "zh"),
                emotion=result.get("emotion", "neutral"),
                text=result.get("text", ""),
                stash=result.get("stash", ""),
            )
        )

    async def _send_transcription_completed(self, result: Dict[str, str]):
        if not self.current_item_id:
            return

        await self._send_event(
            create_transcription_completed_event(
                item_id=self.current_item_id,
                content_index=0,
                language=result.get("language", "zh"),
                emotion=result.get("emotion", "neutral"),
                transcript=result.get("transcript", ""),
            )
        )

    async def _send_event(self, event: Dict[str, Any]):
        try:
            await self.websocket.send_json(event)
            logger.debug(f"Sent event: {event.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def _send_error(
        self,
        error_type: str,
        code: str,
        message: str,
        param: Optional[str] = None,
        event_id: Optional[str] = None,
    ):
        await self._send_event(create_error_event(error_type, code, message, param, event_id))

    async def _cleanup(self):
        if self.asr_session:
            await self.asr_session.close()
        logger.info(f"Session cleaned up: {self.session_id}")
