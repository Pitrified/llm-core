"""Dedup-aware Chroma vector store.

``DeduplicatingMixin`` provides SHA-256-based deduplication that can be mixed
in with any LangChain ``VectorStore`` backend. ``CChroma`` is the concrete
combination of the mixin with Chroma.

Example:
    ::

        from llm_core.vectorstores.cchroma import CChroma

        store = CChroma(collection_name="docs")
        store.add_documents([doc1, doc2])  # duplicates are skipped
"""

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from llm_core.vectorstores.hasher import document_id


class DeduplicatingMixin:
    """Mixin that adds SHA-256-based deduplication to any LangChain VectorStore.

    Place before the concrete backend in the MRO so cooperative ``super()``
    calls forward correctly to the backend's ``add_documents()``.

    The backend must accept ``ids`` as a keyword argument to ``add_documents()``.
    """

    def add_documents(
        self,
        documents: list[Document],
        **kwargs: Any,  # noqa: ANN401
    ) -> list[str]:
        """Add documents, skipping any whose content+metadata hash already exists.

        Args:
            documents: Documents to add.
            kwargs: Forwarded to the backend's ``add_documents``.

        Returns:
            List of IDs for newly added documents (empty if all were duplicates).
        """
        ids = [document_id(doc.page_content, doc.metadata) for doc in documents]
        existing = self._get_existing_ids(ids)
        new_pairs = [
            (doc, id_)
            for doc, id_ in zip(documents, ids, strict=True)
            if id_ not in existing
        ]
        if not new_pairs:
            return []
        new_docs, new_ids = zip(*new_pairs, strict=True)
        return super().add_documents(  # type: ignore[misc]
            documents=list(new_docs),
            ids=list(new_ids),
            **kwargs,
        )

    def _get_existing_ids(self, ids: list[str]) -> frozenset[str]:
        """Return the subset of *ids* already present in the store.

        The default implementation uses ``.get(ids=ids)`` which works for
        Chroma. Backends that do not support ``.get()`` should override this
        method.

        Args:
            ids: Candidate document IDs to check.

        Returns:
            Frozen set of IDs that already exist in the backend.
        """
        result = self.get(ids=ids, include=[])  # type: ignore[attr-defined]
        return frozenset(result["ids"])


class CChroma(DeduplicatingMixin, Chroma):
    """Dedup-aware Chroma vector store.

    MRO: CChroma -> DeduplicatingMixin -> Chroma -> VectorStore

    ``add_documents()`` from the mixin intercepts each call, hashes every
    document via SHA-256, filters out duplicates, then forwards only new
    documents to ``Chroma.add_documents()`` via ``super()``.
    """
