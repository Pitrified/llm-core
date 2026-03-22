"""Local transcription provider using openai-whisper."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from loguru import logger as lg

from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.base import TranscriptionSegment

if TYPE_CHECKING:
    from llm_core.transcription.config.whisper import WhisperConfig


class WhisperTranscriber:
    """Local transcription via openai-whisper.

    The model is loaded eagerly in ``__init__`` so the first ``transcribe()``
    call is fast. Requires the ``whisper`` optional extra to be installed.

    Attributes:
        provider_name: Always "whisper".
    """

    provider_name: str = "whisper"

    def __init__(self, config: WhisperConfig) -> None:
        """Load the Whisper model defined by *config*.

        Args:
            config: WhisperConfig specifying model size, device, and fp16.

        Raises:
            ImportError: If openai-whisper is not installed.
        """
        import whisper  # type: ignore[import-not-found]  # noqa: PLC0415

        lg.info(f"Loading Whisper model '{config.model}' on {config.device}...")
        self._config = config
        self._model = whisper.load_model(config.model, device=config.device)
        lg.info("Whisper model loaded.")

    def transcribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file using the loaded Whisper model.

        Args:
            audio_fp: Path to the audio file.
            language: BCP-47 language hint. Overrides the config default.
                None means use the config default (which may be None for
                auto-detect).
            **kwargs: Additional keyword arguments forwarded to
                ``whisper.model.transcribe()``.

        Returns:
            A TranscriptionResult with text, detected language, and segments.
        """
        effective_language = language or self._config.language
        result = self._model.transcribe(
            str(audio_fp),
            language=effective_language,
            fp16=self._config.fp16,
            **kwargs,
        )
        segments = [
            TranscriptionSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
            )
            for s in result.get("segments", [])
        ]
        return TranscriptionResult(
            text=result["text"],
            language=result.get("language"),
            segments=segments,
            provider=self.provider_name,
        )

    async def atranscribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file asynchronously.

        Wraps the synchronous ``transcribe`` call in ``asyncio.to_thread``
        so the event loop is not blocked during model inference.

        Args:
            audio_fp: Path to the audio file.
            language: BCP-47 language hint. Overrides the config default.
            **kwargs: Additional keyword arguments forwarded to
                ``whisper.model.transcribe()``.

        Returns:
            A TranscriptionResult with text, detected language, and segments.
        """
        return await asyncio.to_thread(
            self.transcribe,
            audio_fp,
            language=language,
            **kwargs,
        )
