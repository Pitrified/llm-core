"""High-level facade for Vectorable entity persistence.

``EntityStore`` accepts ``Vectorable`` entities, converts them to LangChain
``Document`` objects, and stores them in a vector store. On retrieval, raw
documents are returned or typed entities are reconstructed via ``from_document``.

Example:
    ::

        store = EntityStore(config=ChromaConfig())
        store.save(my_entity)
        docs = store.search("find something similar")
        typed = store.search("find something", entity_type=MyEntity)
"""

from collections.abc import Sequence
from typing import Any
from typing import overload

from langchain_core.documents import Document

from llm_core.vectorstores.config.base import VectorStoreConfig
from llm_core.vectorstores.vectorable import Vectorable


class EntityStore:
    """Thin facade that speaks in ``Vectorable`` entities.

    Entities themselves have zero knowledge of this class. The store is the
    only place that holds a reference to the vector DB and performs I/O.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialise the store by creating a vector store from *config*.

        Args:
            config: Vector store configuration used to create the backend.
        """
        self._store = config.create_store()

    # -- write ---------------------------------------------------------------

    def save(self, entity: Vectorable) -> None:
        """Convert *entity* to a Document and upsert it.

        Args:
            entity: A ``Vectorable`` entity to persist.
        """
        doc = entity.to_document()
        self._store.add_documents([doc])

    def save_many(self, entities: Sequence[Vectorable]) -> None:
        """Convert multiple entities to Documents and upsert them.

        Args:
            entities: Sequence of ``Vectorable`` entities to persist.
        """
        docs = [e.to_document() for e in entities]
        self._store.add_documents(docs)

    # -- read ----------------------------------------------------------------

    @overload
    def search(self, query: str, *, k: int = ...) -> list[Document]: ...
    @overload
    def search[T: Vectorable](
        self,
        query: str,
        *,
        entity_type: type[T],
        k: int = ...,
        **filter_kwargs: Any,  # noqa: ANN401
    ) -> list[T]: ...

    def search(
        self,
        query: str,
        *,
        entity_type: type[Vectorable] | None = None,
        k: int = 4,
        **filter_kwargs: Any,
    ) -> Any:
        """Similarity search returning raw Documents or typed entities.

        Args:
            query: Free-text similarity query.
            entity_type: When provided, results are deserialized into this
                type via ``from_document``. Filter kwargs are also forwarded
                to the backend.
            k: Maximum number of results to return.
            filter_kwargs: Additional metadata filters passed to the backend's
                ``similarity_search(filter=...)``.

        Returns:
            Up to *k* results: ``list[Document]`` when ``entity_type`` is
            ``None``, or ``list[T]`` when a type is given.
        """
        search_filter = filter_kwargs or None
        docs = self._store.similarity_search(query, k=k, filter=search_filter)
        if entity_type is None:
            return docs
        return [entity_type.from_document(doc) for doc in docs]
