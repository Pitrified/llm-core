"""Transcription config submodule."""

from llm_core.transcription.config.base import TranscriptionConfig
from llm_core.transcription.config.faster_whisper import FasterWhisperConfig
from llm_core.transcription.config.openai_api import OpenAIAPIConfig
from llm_core.transcription.config.whisper import WhisperConfig

__all__ = [
    "FasterWhisperConfig",
    "OpenAIAPIConfig",
    "TranscriptionConfig",
    "WhisperConfig",
]
