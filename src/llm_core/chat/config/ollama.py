"""Ollama chat configuration."""

from langchain_core.utils.utils import from_env
from pydantic import Field

from llm_core.chat.config.base import ChatConfig


class OllamaChatConfig(ChatConfig):
    """Configuration for a locally-running Ollama chat model.

    Attributes:
        model: Ollama model tag to use. Defaults to "llama3.2".
        model_provider: Provider name for init_chat_model. Always "ollama".
        base_url: Ollama server URL. Reads OLLAMA_BASE_URL env var; defaults to
            None (langchain uses http://localhost:11434).
        max_tokens: Maximum number of tokens to generate. Defaults to 1024.
    """

    model: str = "llama3.2"
    model_provider: str = "ollama"
    base_url: str | None = Field(
        default_factory=from_env("OLLAMA_BASE_URL", default=None)
    )
    max_tokens: int = 1024
