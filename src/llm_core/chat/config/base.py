"""Base class for chat model configuration.

Leverage `init_chat_model`:
https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from llm_core.data_models.basemodel_kwargs import BaseModelKwargs


class ChatConfig(BaseModelKwargs):
    """Base config for a chat model, usable with langchain's init_chat_model.

    Attributes:
        model: Model name (e.g. "gpt-4o-mini").
        model_provider: Provider string dispatched by LangChain
            (e.g. "openai", "ollama").
        temperature: Sampling temperature. Defaults to 0.2.
    """

    model: str
    model_provider: str
    temperature: float = 0.2

    def create_chat_model(self) -> BaseChatModel:
        """Instantiate the chat model from the config.

        Returns:
            A `BaseChatModel` instance constructed via `init_chat_model`.
        """
        return init_chat_model(**self.to_kw(exclude_none=True))
