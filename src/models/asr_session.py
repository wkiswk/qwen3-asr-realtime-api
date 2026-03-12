import asyncio
from typing import Dict, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ASRSession:
    """
    Streaming ASR session for a single WebSocket connection.

    Wraps Qwen3ASRModel streaming inference with proper state management.
    """

    def __init__(
        self,
        asr_manager,
        language: Optional[str] = None,
        sample_rate: int = 16000,
        chunk_size_sec: float = 2.0,
    ):
        self.asr_manager = asr_manager
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_size_sec = chunk_size_sec

        self.state = None
        self._lock = asyncio.Lock()

        self._current_text = ""
        self._current_language = language or ""
        self._emotion = "neutral"
        self._call_count = 0

    async def initialize(self):
        """Initialize streaming state."""
        if not self.asr_manager.is_ready():
            raise RuntimeError("ASR model not loaded")

        try:
            self.state = await self.asr_manager.init_streaming_state(
                context="",
                language=self.language,
                unfixed_chunk_num=2,
                unfixed_token_num=5,
                chunk_size_sec=self.chunk_size_sec,
            )
            self._call_count = 0
            logger.info(
                f"ASR session initialized (language={self.language}, "
                f"chunk_size_sec={self.chunk_size_sec})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ASR session: {e}")
            raise

    async def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk and perform streaming transcription.

        Args:
            audio_chunk: Audio samples (any sample rate, will be converted to 16kHz)
        """
        if self.state is None:
            await self.initialize()

        async with self._lock:
            try:
                pcm16k = self._ensure_16k_mono(audio_chunk)

                self.state = await self.asr_manager.streaming_transcribe(
                    pcm16k=pcm16k, state=self.state
                )

                self._call_count += 1
                self._current_text = self.state.text
                self._current_language = self.state.language

                logger.debug(
                    f"[call {self._call_count:03d}] "
                    f"language={self._current_language!r} "
                    f"text={self._current_text!r}"
                )

            except Exception as e:
                logger.error(f"Error in streaming transcribe: {e}")

    async def get_interim_result(self) -> Optional[Dict[str, str]]:
        """
        Get interim transcription result.

        Returns dict with:
            - language: detected language code
            - emotion: emotion label (always "neutral" for now)
            - text: confirmed text (won't change)
            - stash: tentative text (may change)
        """
        if not self._current_text:
            return None

        text = self._current_text

        if len(text) > 20:
            confirmed_end = len(text) - min(10, len(text) // 3)
            confirmed_text = text[:confirmed_end]
            stash_text = text[confirmed_end:]
        else:
            confirmed_text = ""
            stash_text = text

        return {
            "language": self._detect_language_code(self._current_language),
            "emotion": self._emotion,
            "text": confirmed_text,
            "stash": stash_text,
        }

    async def finish(self) -> Dict[str, str]:
        """
        Finish streaming and return final transcription.

        Returns dict with:
            - transcript: final transcription text
            - language: detected language code
            - emotion: emotion label
        """
        if self.state is None:
            return {"transcript": "", "language": "zh", "emotion": "neutral"}

        async with self._lock:
            try:
                self.state = await self.asr_manager.finish_streaming_transcribe(
                    state=self.state
                )

                logger.info(f"[final] language={self.state.language!r} text_len={len(self.state.text)}")

                return {
                    "transcript": self.state.text,
                    "language": self._detect_language_code(self.state.language),
                    "emotion": self._emotion,
                }
            except Exception as e:
                logger.error(f"Error finishing transcription: {e}")
                return {
                    "transcript": self.state.text if self.state else "",
                    "language": "zh",
                    "emotion": "neutral",
                }

    async def reset(self):
        """Reset session state for new utterance."""
        async with self._lock:
            self.state = None
            self._current_text = ""
            self._current_language = self.language or ""
            self._call_count = 0
            logger.debug("ASR session reset")

    async def close(self):
        """Close session and cleanup."""
        self.state = None
        logger.debug("ASR session closed")

    def _ensure_16k_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure audio is 16kHz mono float32.

        Args:
            audio: Input audio array

        Returns:
            16kHz mono float32 numpy array
        """
        x = np.asarray(audio)

        if x.ndim > 1:
            x = x.mean(axis=1)

        x = x.reshape(-1)

        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        return x

    def _detect_language_code(self, lang: str) -> str:
        """Convert full language name to ISO code."""
        lang_map = {
            "chinese": "zh",
            "english": "en",
            "japanese": "ja",
            "korean": "ko",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "russian": "ru",
            "italian": "it",
            "portuguese": "pt",
            "arabic": "ar",
            "hindi": "hi",
            "indonesian": "id",
            "thai": "th",
            "turkish": "tr",
            "ukrainian": "uk",
            "vietnamese": "vi",
            "czech": "cs",
            "danish": "da",
            "filipino": "fil",
            "finnish": "fi",
            "icelandic": "is",
            "malay": "ms",
            "norwegian": "no",
            "polish": "pl",
            "swedish": "sv",
            "cantonese": "yue",
        }

        lang_lower = lang.lower() if lang else ""
        return lang_map.get(lang_lower, "zh")
