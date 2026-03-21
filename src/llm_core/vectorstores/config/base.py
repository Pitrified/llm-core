"""Abstract base class for vector store configuration."""

from abc import ABC
from abc import abstractmethod

from langchain_core.vectorstores import VectorStore

from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
from llm_core.embeddings.config.base import EmbeddingsConfig


class VectorStoreConfig(BaseModelKwargs, ABC):
    """Abstract base config for a vector store.

    Subclasses must implement ``create_store`` to return a live
    ``VectorStore`` instance backed by the provider of their choice.

    Attributes:
        collection_name: Name of the collection inside the vector store.
        embeddings_config: Embeddings model config forwarded to the store.
    """

    collection_name: str = "default"
    embeddings_config: EmbeddingsConfig | None = None

    @abstractmethod
    def create_store(self) -> VectorStore:
        """Instantiate and return a live vector store from this config."""
        ...
