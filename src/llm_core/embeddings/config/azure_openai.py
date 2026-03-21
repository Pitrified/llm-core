"""Azure OpenAI embeddings configuration."""

from langchain_core.utils.utils import from_env
from langchain_core.utils.utils import secret_from_env
from pydantic import Field
from pydantic import SecretStr

from llm_core.embeddings.config.base import EmbeddingsConfig


class AzureOpenAIEmbeddingsConfig(EmbeddingsConfig):
    """Configuration for Azure OpenAI embedding models.

    Attributes:
        model: Azure deployment model name. Defaults to "text-embedding-3-small".
        provider: Provider name for init_embeddings. Always "azure_openai".
        api_key: Azure OpenAI API key. Reads AZURE_OPENAI_API_KEY env var by default.
        azure_endpoint: Azure OpenAI endpoint. Reads AZURE_OPENAI_ENDPOINT env var by
            default.
        api_version: Azure OpenAI API version. Defaults to "2024-02-01".
    """

    model: str = "text-embedding-3-small"
    provider: str = "azure_openai"
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("AZURE_OPENAI_API_KEY", default=None)
    )
    azure_endpoint: str | None = Field(
        default_factory=from_env("AZURE_OPENAI_ENDPOINT", default=None)
    )
    api_version: str = "2024-02-01"
