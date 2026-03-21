"""HuggingFace chat configuration."""

from langchain_core.utils.utils import secret_from_env
from pydantic import Field
from pydantic import SecretStr

from llm_core.chat.config.base import ChatConfig


class HuggingFaceChatConfig(ChatConfig):
    """Configuration for the HuggingFace chat model (via Inference API).

    Attributes:
        model: HuggingFace model ID to use. Defaults to
            "HuggingFaceTB/SmolLM2-135M-Instruct".
        model_provider: Provider name for init_chat_model. Always "huggingface".
        api_key: HuggingFace API token. Reads HUGGINGFACEHUB_API_TOKEN env var by
            default.
        max_tokens: Maximum number of tokens to generate. Defaults to 1024.
    """

    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_provider: str = "huggingface"
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("HUGGINGFACEHUB_API_TOKEN", default=None)
    )
    max_tokens: int = 1024
