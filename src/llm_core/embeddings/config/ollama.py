"""Ollama embeddings configuration."""

from langchain_core.utils.utils import from_env
from pydantic import Field

from llm_core.embeddings.config.base import EmbeddingsConfig


class OllamaEmbeddingsConfig(EmbeddingsConfig):
    """Configuration for a locally-running Ollama embedding model.

    Attributes:
        model: Ollama embedding model tag to use. Defaults to "nomic-embed-text".
        provider: Provider name for init_embeddings. Always "ollama".
        base_url: Ollama server URL. Reads OLLAMA_BASE_URL env var; defaults to
            None (langchain uses http://localhost:11434).
    """

    model: str = "nomic-embed-text"
    provider: str = "ollama"
    base_url: str | None = Field(
        default_factory=from_env("OLLAMA_BASE_URL", default=None)
    )
