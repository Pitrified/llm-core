"""Core types for the transcription module.

Defines the structured result objects (TranscriptionResult, TranscriptionSegment)
and the BaseTranscriber protocol that all provider implementations must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class TranscriptionSegment:
    """One timed segment of a transcription.

    Not all providers populate this; treat as best-effort.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        text: Transcript text for this segment.
    """

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Output of a transcription call.

    Attributes:
        text: Full transcript as a single string. Always present.
        language: BCP-47 language code detected or specified (e.g. "en", "it").
            None if the provider did not detect or report it.
        segments: Word- or sentence-level segments with timestamps.
            Empty list if the provider does not support segmentation.
        provider: Provider identifier string, for logging and traceability.
    """

    text: str
    language: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    provider: str = ""


@runtime_checkable
class BaseTranscriber(Protocol):
    """Common interface for all transcription backends.

    Both sync and async surfaces are required. For sync-only backends
    (openai-whisper, faster-whisper), ``atranscribe`` wraps ``transcribe``
    in ``asyncio.to_thread`` - this wrapping lives inside the provider, not
    in the caller.

    The ``language`` parameter is an explicit keyword argument (not buried in
    ``**kwargs``) because it is the single most common per-call override.

    Attributes:
        provider_name: Short identifier string for the provider (e.g. "whisper").
    """

    provider_name: str

    def transcribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file synchronously.

        Args:
            audio_fp: Path to the audio file. Format conversion is the
                caller's responsibility - pass a file supported by the
                underlying provider.
            language: BCP-47 language hint (e.g. "en", "it"). Overrides the
                config-level default. None means use the config default, which
                may itself be None (auto-detect).
            **kwargs: Provider-specific keyword arguments forwarded to the
                underlying model.

        Returns:
            A TranscriptionResult with at least ``text`` populated.
        """
        ...

    async def atranscribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Transcribe an audio file asynchronously.

        For CPU-bound backends (openai-whisper, faster-whisper), this wraps
        the sync ``transcribe`` call in ``asyncio.to_thread`` internally.

        Args:
            audio_fp: Path to the audio file.
            language: BCP-47 language hint. Overrides the config-level default.
            **kwargs: Provider-specific keyword arguments.

        Returns:
            A TranscriptionResult with at least ``text`` populated.
        """
        ...
