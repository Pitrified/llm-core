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

from langchain_chroma import Chroma

from llm_core.vectorstores.mixins.deduplicating import DeduplicatingMixin


class CChroma(DeduplicatingMixin, Chroma):
    """Dedup-aware Chroma vector store.

    MRO: CChroma -> DeduplicatingMixin -> Chroma -> VectorStore

    ``add_documents()`` from the mixin intercepts each call, hashes every
    document via SHA-256, filters out duplicates, then forwards only new
    documents to ``Chroma.add_documents()`` via ``super()``.
    """
