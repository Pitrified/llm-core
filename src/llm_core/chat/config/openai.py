"""OpenAI chat configuration."""

from langchain_core.utils.utils import secret_from_env
from pydantic import Field
from pydantic import SecretStr

from llm_core.chat.config.base import ChatConfig


class ChatOpenAIConfig(ChatConfig):
    """Configuration for the OpenAI chat model.

    Attributes:
        model: Model name to use. Defaults to "gpt-4o-mini".
        model_provider: Provider name for init_chat_model. Always "openai".
        api_key: OpenAI API key. Reads OPENAI_API_KEY env var by default.
    """

    model: str = "gpt-4o-mini"
    model_provider: str = "openai"
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
