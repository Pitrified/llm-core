"""Local transcription provider using faster-whisper (CTranslate2 backend)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from loguru import logger as lg

from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.base import TranscriptionSegment

if TYPE_CHECKING:
    from llm_core.transcription.config.faster_whisper import FasterWhisperConfig


class FasterWhisperTranscriber:
    """Local transcription via faster-whisper (CTranslate2 backend).

    Substantially faster inference than openai-whisper on the same model size,
    with a lighter dependency footprint (no torch required).

    The model is loaded eagerly in ``__init__``. Requires the
    ``faster-whisper`` optional extra to be installed.

    Attributes:
        provider_name: Always "faster_whisper".
    """

    provider_name: str = "faster_whisper"

    def __init__(self, config: FasterWhisperConfig) -> None:
        """Load the faster-whisper model defined by *config*.

        Args:
            config: FasterWhisperConfig specifying model, device, and
                compute_type.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]  # noqa: I001,PLC0415

        lg.info(
            f"Loading faster-whisper model '{config.model}' on {config.device} "
            f"with compute_type='{config.compute_type}'..."
        )
        self._config = config
        self._model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
        lg.info("faster-whisper model loaded.")

    def transcribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file using the loaded faster-whisper model.

        The segment generator is consumed eagerly to materialise both the
        segment list and the full transcript text in a single pass.

        Args:
            audio_fp: Path to the audio file.
            language: BCP-47 language hint. Overrides the config default.
                None means use the config default (which may be None for
                auto-detect).
            **kwargs: Additional keyword arguments forwarded to
                ``WhisperModel.transcribe()``.

        Returns:
            A TranscriptionResult with text, detected language, and segments.
        """
        effective_language = language or self._config.language
        segments_gen, info = self._model.transcribe(
            str(audio_fp),
            language=effective_language,
            beam_size=self._config.beam_size,
            **kwargs,
        )
        segments = [
            TranscriptionSegment(start=s.start, end=s.end, text=s.text)
            for s in segments_gen
        ]
        full_text = " ".join(s.text.strip() for s in segments)
        return TranscriptionResult(
            text=full_text,
            language=info.language,
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
                ``WhisperModel.transcribe()``.

        Returns:
            A TranscriptionResult with text, detected language, and segments.
        """
        return await asyncio.to_thread(
            self.transcribe,
            audio_fp,
            language=language,
            **kwargs,
        )
