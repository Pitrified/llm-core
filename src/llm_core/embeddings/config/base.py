"""Base class for embeddings model configuration.

Leverage `init_embeddings`:
https://reference.langchain.com/python/langchain/embeddings/base/init_embeddings
"""

from langchain.embeddings import Embeddings
from langchain.embeddings import init_embeddings

from llm_core.data_models.basemodel_kwargs import BaseModelKwargs


class EmbeddingsConfig(BaseModelKwargs):
    """Base config for an embeddings model, usable with langchain's init_embeddings.

    Fields should match the keywords expected by langchain's init function so
    `to_kw()` can be passed directly.

    Attributes:
        model: Model name (e.g. "text-embedding-3-small").
        provider: Provider string dispatched by LangChain
            (e.g. "openai", "ollama").
    """

    model: str
    provider: str

    def create_embeddings(self) -> Embeddings:
        """Instantiate the embeddings model from the config.

        Returns:
            An `Embeddings` instance constructed via `init_embeddings`.
        """
        return init_embeddings(**self.to_kw(exclude_none=True))
