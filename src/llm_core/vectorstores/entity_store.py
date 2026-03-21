"""High-level facade for Vectorable entity persistence.

``EntityStore`` accepts ``Vectorable`` entities, converts them to LangChain
``Document`` objects, and stores them in a vector store. On retrieval, raw
documents are returned or typed entities are reconstructed via ``from_document``.

Filtering is delegated entirely to the underlying store via the
``CondSearchable`` protocol - ``EntityStore`` never touches serialization.

Example:
    ::

        store = EntityStore(config=ChromaConfig())
        store.save(my_entity)
        docs = store.search("find something similar")
        typed = store.search("find something", entity_type=MyEntity)
        filtered = store.search(
            "find active",
            cond=CompCond("status", CompOp.EQ, "active"),
        )
"""

from collections.abc import Iterable
from typing import Any
from typing import overload

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from llm_core.vectorstores.cond import AnyCond
from llm_core.vectorstores.cond import CondSearchable
from llm_core.vectorstores.config.base import VectorStoreConfig
from llm_core.vectorstores.vectorable import Vectorable


class EntityStore:
    """Thin facade that speaks in ``Vectorable`` entities.

    Entities themselves have zero knowledge of this class. The store is the
    only place that holds a reference to the vector DB and performs I/O.
    """

    _vs: VectorStore
    _cs: CondSearchable

    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialise the store by creating a vector store from *config*.

        The backend must satisfy both ``VectorStore`` (for add/save) and
        ``CondSearchable`` (for filtered search).

        Args:
            config: Vector store configuration used to create the backend.
        """
        store = config.create_store()
        if not isinstance(store, CondSearchable):
            msg = (
                f"{type(store).__name__} does not satisfy CondSearchable; "
                "EntityStore requires a backend with cond_search() support"
            )
            raise TypeError(msg)
        self._vs = store
        self._cs = store

    # -- write ---------------------------------------------------------------

    def save(self, entities: Vectorable | Iterable[Vectorable]) -> None:
        """Convert one or more entities to Documents and upsert them.

        Args:
            entities: One or an iterable of ``Vectorable`` entities to persist.
        """
        if isinstance(entities, Vectorable):
            entities = [entities]
        docs = [e.to_document() for e in entities]
        if len(docs) == 0:
            return
        self._vs.add_documents(docs)

    # -- read ----------------------------------------------------------------

    @overload
    def search(
        self,
        query: str,
        *,
        entity_type: None = None,
        k: int = ...,
        cond: AnyCond | None = ...,
    ) -> list[Document]: ...
    @overload
    def search[T: Vectorable](
        self,
        query: str,
        *,
        entity_type: type[T],
        k: int = ...,
        cond: AnyCond | None = ...,
    ) -> list[T]: ...

    def search(
        self,
        query: str,
        *,
        entity_type: type[Vectorable] | None = None,
        k: int = 4,
        cond: AnyCond | None = None,
    ) -> Any:
        """Similarity search returning raw Documents or typed entities.

        Args:
            query: Free-text similarity query.
            entity_type: When provided, results are deserialized into this
                type via ``from_document``.
            k: Maximum number of results to return.
            cond: Optional condition tree for metadata/document-level
                filtering. Serialization is handled by the backend adapter.

        Returns:
            Up to *k* results: ``list[Document]`` when ``entity_type`` is
            ``None``, or ``list[T]`` when a type is given.
        """
        docs = self._cs.cond_search(query, k=k, cond=cond)
        if entity_type is None:
            return docs
        return [entity_type.from_document(doc) for doc in docs]
