"""Tests for CChroma (DeduplicatingMixin + Chroma)."""

from typing import TYPE_CHECKING
import uuid

from langchain_core.documents import Document
import pytest

try:
    import chromadb  # noqa: F401

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

if TYPE_CHECKING:
    from llm_core.vectorstores.cchroma import CChroma

pytestmark = pytest.mark.skipif(not HAS_CHROMA, reason="chromadb not installed")


def _make_store() -> "CChroma":
    from llm_core.vectorstores.cchroma import CChroma  # noqa: PLC0415

    return CChroma(collection_name=f"test_{uuid.uuid4().hex[:8]}")


class TestCChromaDedup:
    """Tests for CChroma deduplication."""

    def test_add_documents_returns_ids(self) -> None:
        """Adding new documents returns their IDs."""
        store = _make_store()
        docs = [
            Document(page_content="doc1", metadata={"entity_type": "test"}),
            Document(page_content="doc2", metadata={"entity_type": "test"}),
        ]
        ids = store.add_documents(docs)
        assert len(ids) == 2

    def test_duplicate_documents_skipped(self) -> None:
        """Adding the same documents twice does not duplicate them."""
        store = _make_store()
        docs = [
            Document(page_content="unique content", metadata={"entity_type": "test"}),
        ]
        first_ids = store.add_documents(docs)
        assert len(first_ids) == 1

        second_ids = store.add_documents(docs)
        assert len(second_ids) == 0

    def test_mixed_new_and_duplicate(self) -> None:
        """Only new documents are added when mixed with duplicates."""
        store = _make_store()
        doc1 = Document(page_content="first", metadata={"entity_type": "test"})
        doc2 = Document(page_content="second", metadata={"entity_type": "test"})

        store.add_documents([doc1])
        ids = store.add_documents([doc1, doc2])
        # Only doc2 is new
        assert len(ids) == 1

    def test_different_metadata_not_duplicate(self) -> None:
        """Same content with different metadata are NOT duplicates."""
        store = _make_store()
        doc1 = Document(page_content="same", metadata={"entity_type": "a"})
        doc2 = Document(page_content="same", metadata={"entity_type": "b"})

        ids1 = store.add_documents([doc1])
        ids2 = store.add_documents([doc2])
        assert len(ids1) == 1
        assert len(ids2) == 1
