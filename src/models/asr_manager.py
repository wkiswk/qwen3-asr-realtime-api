import asyncio
import os
import warnings
from typing import Any, Dict, Optional

import torch

warnings.filterwarnings("ignore", message="Casting torch.bfloat16 to torch.float16")
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32")
warnings.filterwarnings("ignore", message="The following generation flags are not valid")
warnings.filterwarnings("ignore", message="We must use the `spawn` multiprocessing start method")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from utils.logger import get_logger

logger = get_logger(__name__)


class ASRManager:
    """
    ASR Model Manager for Qwen3-ASR with vLLM backend.

    Note: Streaming inference is only available with vLLM backend.
    """

    def __init__(self):
        self.model = None
        self.model_path: str = os.getenv("QWEN3_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
        self.gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.5"))
        self.max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "64"))
        self.dtype: str = os.getenv("MODEL_DTYPE", "auto")
        # 添加全局锁，确保模型推理的线程安全
        self._lock = asyncio.Lock()

    async def load_model(self):
        """
        Load Qwen3-ASR model with vLLM backend for streaming inference.
        """
        logger.info(
            f"vLLM config: gpu_memory_utilization={self.gpu_memory_utilization}, "
            f"max_new_tokens={self.max_new_tokens}, dtype={self.dtype}"
        )

        try:
            from qwen_asr import Qwen3ASRModel

            self.model = await asyncio.to_thread(
                Qwen3ASRModel.LLM,
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_new_tokens=self.max_new_tokens,
                dtype=self.dtype,
                max_model_len=32768,
            )

            logger.info("Model loaded successfully with vLLM backend")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self):
        """Unload model and free GPU memory."""
        if self.model:
            # Properly shutdown vLLM engine to release GPU resources
            # Qwen3ASRModel.model is the raw vllm.LLM instance
            vllm_llm = getattr(self.model, "model", None)
            if vllm_llm is not None:
                # vLLM LLM has llm_engine -> engine_core with shutdown()
                llm_engine = getattr(vllm_llm, "llm_engine", None)
                if llm_engine is not None:
                    engine_core = getattr(llm_engine, "engine_core", None)
                    if engine_core is not None and hasattr(engine_core, "shutdown"):
                        try:
                            logger.info("Shutting down vLLM engine core...")
                            engine_core.shutdown()
                            logger.info("vLLM engine core shutdown complete")
                        except Exception as e:
                            logger.warning(f"Error during engine core shutdown: {e}")

            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("Model unloaded")

    def get_model(self):
        return self.model

    def is_ready(self) -> bool:
        return self.model is not None

    async def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ):
        """
        Initialize streaming ASR state.

        Args:
            context: Context text for the ASR session
            language: Optional language hint (e.g., "Chinese", "English")
            unfixed_chunk_num: Number of initial chunks without prefix
            unfixed_token_num: Number of tokens to rollback for prefix
            chunk_size_sec: Audio chunk size in seconds
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        async with self._lock:
            return await asyncio.to_thread(
                self.model.init_streaming_state,
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )

    async def streaming_transcribe(self, pcm16k: Any, state: Any) -> Any:
        """
        Perform streaming transcription on audio chunk.

        Args:
            pcm16k: 16kHz mono PCM audio (numpy array)
            state: Streaming state object

        Returns:
            Updated state object
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        async with self._lock:
            return await asyncio.to_thread(
                self.model.streaming_transcribe, pcm16k, state
            )

    async def finish_streaming_transcribe(self, state: Any) -> Any:
        """
        Finish streaming transcription and process remaining audio.

        Args:
            state: Streaming state object

        Returns:
            Final state with complete transcription
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        async with self._lock:
            return await asyncio.to_thread(
                self.model.finish_streaming_transcribe, state
            )
