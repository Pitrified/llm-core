"""Transcription module for llm-core.

Provides a provider-agnostic transcription interface with support for
local openai-whisper, faster-whisper, and the remote OpenAI Whisper API.

Pattern mirrors the ``chat`` and ``embeddings`` modules: a base config with a
``create_transcriber()`` factory, provider-specific subclasses, and a fake
implementation for tests.

Example:
    ::

        from pathlib import Path

        from llm_core.transcription.base import TranscriptionResult
        from llm_core.transcription.testing.fake import FakeTranscriberConfig

        config = FakeTranscriberConfig(
            responses=[TranscriptionResult(text="Hello.", language="en")]
        )
        transcriber = config.create_transcriber()
        result = transcriber.transcribe(Path("audio.mp3"))
"""

from llm_core.transcription.base import BaseTranscriber
from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.base import TranscriptionSegment
from llm_core.transcription.config.base import TranscriptionConfig
from llm_core.transcription.config.faster_whisper import FasterWhisperConfig
from llm_core.transcription.config.openai_api import OpenAIAPIConfig
from llm_core.transcription.config.whisper import WhisperConfig

__all__ = [
    "BaseTranscriber",
    "FasterWhisperConfig",
    "OpenAIAPIConfig",
    "TranscriptionConfig",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperConfig",
]
