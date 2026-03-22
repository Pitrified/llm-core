"""Base class for transcription model configuration."""

from abc import abstractmethod

from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
from llm_core.transcription.base import BaseTranscriber


class TranscriptionConfig(BaseModelKwargs):
    """Base config for a transcription model.

    Subclasses must implement ``create_transcriber()`` to instantiate the
    concrete provider. Fields common to all providers (model identifier and
    default language) live here; provider-specific fields are added in
    subclasses.

    The call-time ``language`` argument on ``transcribe()`` takes precedence
    over the config-level ``language`` default. When both are ``None``,
    auto-detection is used (where the provider supports it).

    Attributes:
        model: Model identifier. Meaning is provider-specific
            (e.g. "medium" for Whisper, "whisper-1" for the OpenAI API).
        language: Default language hint (BCP-47, e.g. "en", "it").
            None means auto-detect (where supported).
    """

    model: str
    language: str | None = None

    @abstractmethod
    def create_transcriber(self) -> BaseTranscriber:
        """Instantiate the transcriber from this config.

        Returns:
            A BaseTranscriber implementation ready to accept audio files.
        """
