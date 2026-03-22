"""Configuration for local openai-whisper transcription."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_core.transcription.config.base import TranscriptionConfig

if TYPE_CHECKING:
    from llm_core.transcription.base import BaseTranscriber


class WhisperConfig(TranscriptionConfig):
    """Config for local openai-whisper.

    Requires the ``whisper`` optional extra: ``pip install llm-core[whisper]``.

    Model is loaded eagerly when ``create_transcriber()`` is called - the first
    transcription call is then fast. Use ``FakeTranscriberConfig`` in tests to
    avoid loading a real model.

    Attributes:
        model: Whisper model size. Defaults to "medium".
            One of: tiny, base, small, medium, large, large-v2, large-v3.
        device: Torch device. "cpu" or "cuda". Defaults to "cpu".
        fp16: Use half-precision. Meaningful only on CUDA. Defaults to False.
    """

    model: str = "medium"
    device: str = "cpu"
    fp16: bool = False

    def create_transcriber(self) -> BaseTranscriber:
        """Instantiate a WhisperTranscriber from this config.

        Returns:
            A WhisperTranscriber with the model loaded and ready.

        Raises:
            ImportError: If openai-whisper is not installed.
        """
        from llm_core.transcription.providers.whisper import WhisperTranscriber  # noqa: I001,PLC0415

        return WhisperTranscriber(config=self)
