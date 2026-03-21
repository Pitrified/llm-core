"""OpenAI embeddings configuration."""

from langchain_core.utils.utils import secret_from_env
from pydantic import Field
from pydantic import SecretStr

from llm_core.embeddings.config.base import EmbeddingsConfig


class OpenAIEmbeddingsConfig(EmbeddingsConfig):
    """Configuration for OpenAI embedding models.

    Attributes:
        model: OpenAI embedding model name. Defaults to "text-embedding-3-small".
        provider: Provider name for init_embeddings. Always "openai".
        api_key: OpenAI API key. Reads OPENAI_API_KEY env var by default.
    """

    model: str = "text-embedding-3-small"
    provider: str = "openai"
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
