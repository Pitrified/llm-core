"""Remote transcription provider using the OpenAI Whisper API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.base import TranscriptionSegment

if TYPE_CHECKING:
    from pathlib import Path

    from llm_core.transcription.config.openai_api import OpenAIAPIConfig


class OpenAIAPITranscriber:
    """Remote transcription via the OpenAI Whisper API.

    No local model is loaded - audio is sent to OpenAI's servers. Suitable
    for lighter deployments where torch or a GPU is not available.

    Requires the ``openai`` optional extra to be installed
    (``pip install llm-core[openai]``).

    Attributes:
        provider_name: Always "openai_api".
    """

    provider_name: str = "openai_api"

    def __init__(self, config: OpenAIAPIConfig) -> None:
        """Store the config for use in transcription calls.

        The openai client is created per-call to avoid holding a long-lived
        connection object.

        Args:
            config: OpenAIAPIConfig with the API key and model settings.

        Raises:
            ImportError: If the openai package is not installed.
        """
        try:
            import openai as _openai  # type: ignore[import-not-found]  # noqa: F401,PLC0415
        except ImportError as exc:
            msg = (
                "openai is not installed. Install it with: pip install llm-core[openai]"
            )
            raise ImportError(msg) from exc
        self._config = config

    def transcribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file via the OpenAI Whisper API.

        Args:
            audio_fp: Path to the audio file. Supported formats: mp3, mp4,
                mpeg, mpga, m4a, wav, webm (OpenAI API limits).
            language: BCP-47 language hint. Overrides the config default.
                None means use the config default (which may be None for
                auto-detect).
            **kwargs: Additional keyword arguments forwarded to
                ``openai.audio.transcriptions.create()``.

        Returns:
            A TranscriptionResult. Segments are populated when
            ``response_format`` is "verbose_json" (the default).
        """
        import openai  # type: ignore[import-not-found]  # noqa: PLC0415

        effective_language = language or self._config.language
        secret = self._config.api_key
        api_key = secret.get_secret_value() if secret is not None else None

        client = openai.OpenAI(api_key=api_key)

        extra: dict[str, object] = dict(kwargs)
        if effective_language is not None:
            extra["language"] = effective_language

        with audio_fp.open("rb") as f:
            response = client.audio.transcriptions.create(
                model=self._config.model,
                file=(audio_fp.name, f, "audio/mpeg"),
                response_format=self._config.response_format,
                **extra,
            )

        return self._parse_response(response)

    async def atranscribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file asynchronously via the OpenAI Whisper API.

        Wraps the synchronous ``transcribe`` call in ``asyncio.to_thread``.
        For high-throughput use cases, consider batching calls instead.

        Args:
            audio_fp: Path to the audio file.
            language: BCP-47 language hint. Overrides the config default.
            **kwargs: Additional keyword arguments forwarded to the API.

        Returns:
            A TranscriptionResult. Segments are populated when
            ``response_format`` is "verbose_json".
        """
        return await asyncio.to_thread(
            self.transcribe,
            audio_fp,
            language=language,
            **kwargs,
        )

    def _parse_response(self, response: object) -> TranscriptionResult:
        """Parse the OpenAI API response into a TranscriptionResult.

        Args:
            response: The response object returned by
                ``openai.audio.transcriptions.create()``.

        Returns:
            A TranscriptionResult with available fields populated.
        """
        text = getattr(response, "text", "") or ""
        language = getattr(response, "language", None)
        raw_segments = getattr(response, "segments", None) or []
        segments = [
            TranscriptionSegment(
                start=float(getattr(s, "start", 0.0)),
                end=float(getattr(s, "end", 0.0)),
                text=str(getattr(s, "text", "")),
            )
            for s in raw_segments
        ]
        return TranscriptionResult(
            text=text,
            language=language,
            segments=segments,
            provider=self.provider_name,
        )
