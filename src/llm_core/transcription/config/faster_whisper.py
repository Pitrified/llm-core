"""Configuration for faster-whisper transcription (CTranslate2 backend)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_core.transcription.config.base import TranscriptionConfig

if TYPE_CHECKING:
    from llm_core.transcription.base import BaseTranscriber


class FasterWhisperConfig(TranscriptionConfig):
    """Config for faster-whisper (CTranslate2 backend).

    Substantially faster inference than openai-whisper on the same model size.
    Uses CTranslate2 quantization instead of Torch - lighter footprint with
    no ``torch`` dependency.

    Requires the ``faster-whisper`` optional extra:
    ``pip install llm-core[faster-whisper]``.

    Attributes:
        model: Model size or HuggingFace model ID. Defaults to "medium".
        device: "cpu" or "cuda". Defaults to "cpu".
        compute_type: CTranslate2 quantization type. "int8" is fastest on CPU,
            "float16" is fastest on GPU. Defaults to "int8".
        beam_size: Beam search width. Higher means more accurate but slower.
            Defaults to 5 (faster-whisper default).
    """

    model: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5

    def create_transcriber(self) -> BaseTranscriber:
        """Instantiate a FasterWhisperTranscriber from this config.

        Returns:
            A FasterWhisperTranscriber with the model loaded and ready.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        from llm_core.transcription.providers.faster_whisper import (  # noqa: PLC0415
            FasterWhisperTranscriber,
        )

        return FasterWhisperTranscriber(config=self)
