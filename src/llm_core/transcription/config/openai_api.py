"""Configuration for the OpenAI Whisper API (remote, cloud-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

from langchain_core.utils.utils import secret_from_env
from pydantic import Field
from pydantic import SecretStr

from llm_core.transcription.config.base import TranscriptionConfig

if TYPE_CHECKING:
    from llm_core.transcription.base import BaseTranscriber


class OpenAIAPIConfig(TranscriptionConfig):
    """Config for the OpenAI Whisper API (remote, no local model).

    Useful for deployments where torch or a GPU is not available. Requires
    an OpenAI API key and a network connection.

    Requires the ``openai`` optional extra: ``pip install llm-core[openai]``.

    Attributes:
        model: Always "whisper-1" for the current OpenAI API.
        api_key: OpenAI API key. Reads OPENAI_API_KEY env var by default.
        response_format: API response format. "verbose_json" populates
            ``segments`` in ``TranscriptionResult``. Defaults to
            "verbose_json".
    """

    model: str = "whisper-1"
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("OPENAI_API_KEY", default=None),
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = (
        "verbose_json"
    )

    def create_transcriber(self) -> BaseTranscriber:
        """Instantiate an OpenAIAPITranscriber from this config.

        Returns:
            An OpenAIAPITranscriber ready to call the OpenAI Whisper API.

        Raises:
            ImportError: If the openai package is not installed.
        """
        from llm_core.transcription.providers.openai_api import OpenAIAPITranscriber  # noqa: I001,PLC0415

        return OpenAIAPITranscriber(config=self)
