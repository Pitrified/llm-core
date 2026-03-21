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


class TestCChromaSerialize:
    """Tests for CChroma._serialize() and related private methods."""

    def test_serialize_none(self) -> None:
        """_serialize(None) returns (None, None)."""
        store = _make_store()
        assert store._serialize(None) == (None, None)  # noqa: SLF001

    def test_serialize_comp_cond(self) -> None:
        """_serialize(CompCond) returns a metadata filter."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415

        store = _make_store()
        cond = CompCond("tag", CompOp.EQ, "x")
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"tag": {"$eq": "x"}}
        assert doc is None

    def test_serialize_comp_cond_gt(self) -> None:
        """_serialize(CompCond GT) returns correct Chroma operator."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415

        store = _make_store()
        cond = CompCond("age", CompOp.GT, 30)
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"age": {"$gt": 30}}
        assert doc is None

    def test_serialize_inclusion_cond(self) -> None:
        """_serialize(InclusionCond) returns inclusion filter."""
        from llm_core.vectorstores.cond import InclusionCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import InclusionOp  # noqa: PLC0415

        store = _make_store()
        cond = InclusionCond("status", InclusionOp.IN, ["a", "b"])
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"status": {"$in": ["a", "b"]}}
        assert doc is None

    def test_serialize_logical_and(self) -> None:
        """_serialize(LogicalCond AND) returns $and filter."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalOp  # noqa: PLC0415

        store = _make_store()
        cond = LogicalCond(
            LogicalOp.AND,
            [CompCond("a", CompOp.EQ, 1), CompCond("b", CompOp.GT, 2)],
        )
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"$and": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}
        assert doc is None

    def test_serialize_logical_or(self) -> None:
        """_serialize(LogicalCond OR) returns $or filter."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalOp  # noqa: PLC0415

        store = _make_store()
        cond = LogicalCond(
            LogicalOp.OR,
            [CompCond("a", CompOp.EQ, 1), CompCond("b", CompOp.EQ, 2)],
        )
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}
        assert doc is None

    def test_serialize_not_cond_eq(self) -> None:
        """_serialize(NotCond(CompCond EQ)) negates to $ne."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        cond = NotCond(CompCond("tag", CompOp.EQ, "x"))
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"tag": {"$ne": "x"}}
        assert doc is None

    def test_serialize_not_cond_gt(self) -> None:
        """_serialize(NotCond(CompCond GT)) negates to $lte."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        cond = NotCond(CompCond("age", CompOp.GT, 30))
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"age": {"$lte": 30}}
        assert doc is None

    def test_serialize_not_inclusion_in(self) -> None:
        """_serialize(NotCond(InclusionCond IN)) negates to $nin."""
        from llm_core.vectorstores.cond import InclusionCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import InclusionOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        cond = NotCond(InclusionCond("status", InclusionOp.IN, ["a", "b"]))
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"status": {"$nin": ["a", "b"]}}
        assert doc is None

    def test_serialize_not_and_de_morgan(self) -> None:
        """_serialize(NotCond(AND)) applies De Morgan: NOT AND -> OR of negations."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        cond = NotCond(
            LogicalCond(
                LogicalOp.AND,
                [CompCond("a", CompOp.EQ, 1), CompCond("b", CompOp.GT, 2)],
            ),
        )
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"$or": [{"a": {"$ne": 1}}, {"b": {"$lte": 2}}]}
        assert doc is None

    def test_serialize_not_or_de_morgan(self) -> None:
        """_serialize(NotCond(OR)) applies De Morgan: NOT OR -> AND of negations."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        cond = NotCond(
            LogicalCond(
                LogicalOp.OR,
                [CompCond("a", CompOp.EQ, 1), CompCond("b", CompOp.EQ, 2)],
            ),
        )
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"$and": [{"a": {"$ne": 1}}, {"b": {"$ne": 2}}]}
        assert doc is None

    def test_serialize_double_not(self) -> None:
        """_serialize(NotCond(NotCond(x))) collapses to x."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        inner = CompCond("tag", CompOp.EQ, "x")
        cond = NotCond(NotCond(inner))
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta == {"tag": {"$eq": "x"}}
        assert doc is None

    def test_serialize_doc_cond_contains(self) -> None:
        """_serialize(DocCond) returns where_document with $contains."""
        from llm_core.vectorstores.cond import DocCond  # noqa: PLC0415

        store = _make_store()
        cond = DocCond("hello")
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta is None
        assert doc == {"$contains": "hello"}

    def test_serialize_doc_cond_not_contains(self) -> None:
        """_serialize(DocCond negate=True) returns $not_contains."""
        from llm_core.vectorstores.cond import DocCond  # noqa: PLC0415

        store = _make_store()
        cond = DocCond("hello", negate=True)
        meta, doc = store._serialize(cond)  # noqa: SLF001
        assert meta is None
        assert doc == {"$not_contains": "hello"}

    def test_serialize_doc_cond_nested_raises(self) -> None:
        """DocCond nested inside LogicalCond raises TypeError."""
        from llm_core.vectorstores.cond import DocCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import LogicalOp  # noqa: PLC0415

        store = _make_store()
        cond = LogicalCond(LogicalOp.AND, [DocCond("hello")])
        with pytest.raises(TypeError, match="DocCond cannot be nested"):
            store._serialize(cond)  # noqa: SLF001


class TestCChromaCondSearch:
    """Integration tests for CChroma.cond_search()."""

    def test_cond_search_no_filter(self) -> None:
        """cond_search with cond=None returns all matching results."""
        store = _make_store()
        store.add_documents(
            [
                Document(page_content="alpha item", metadata={"tag": "a"}),
                Document(page_content="beta item", metadata={"tag": "b"}),
            ]
        )
        results = store.cond_search("item", k=5)
        assert len(results) == 2

    def test_cond_search_with_comp_eq(self) -> None:
        """cond_search with CompCond EQ filters correctly."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415

        store = _make_store()
        store.add_documents(
            [
                Document(page_content="alpha item", metadata={"tag": "a"}),
                Document(page_content="beta item", metadata={"tag": "b"}),
            ]
        )
        results = store.cond_search(
            "item",
            k=5,
            cond=CompCond("tag", CompOp.EQ, "a"),
        )
        assert len(results) == 1
        assert results[0].page_content == "alpha item"

    def test_cond_search_with_not_cond(self) -> None:
        """cond_search with NotCond excludes matching documents."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415
        from llm_core.vectorstores.cond import NotCond  # noqa: PLC0415

        store = _make_store()
        store.add_documents(
            [
                Document(page_content="alpha item", metadata={"tag": "a"}),
                Document(page_content="beta item", metadata={"tag": "b"}),
            ]
        )
        results = store.cond_search(
            "item",
            k=5,
            cond=NotCond(CompCond("tag", CompOp.EQ, "a")),
        )
        assert len(results) == 1
        assert results[0].page_content == "beta item"

    def test_cond_search_with_logical_and(self) -> None:
        """cond_search with AND combines filters."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415

        store = _make_store()
        store.add_documents(
            [
                Document(
                    page_content="alpha one",
                    metadata={"tag": "a", "priority": "high"},
                ),
                Document(
                    page_content="alpha two",
                    metadata={"tag": "a", "priority": "low"},
                ),
                Document(
                    page_content="beta one",
                    metadata={"tag": "b", "priority": "high"},
                ),
            ]
        )
        cond = CompCond("tag", CompOp.EQ, "a") & CompCond(
            "priority",
            CompOp.EQ,
            "high",
        )
        results = store.cond_search("alpha", k=5, cond=cond)
        assert len(results) == 1
        assert results[0].page_content == "alpha one"

    def test_cond_search_with_doc_cond(self) -> None:
        """cond_search with DocCond filters on document content."""
        store = _make_store()
        store.add_documents(
            [
                Document(page_content="recipe for pasta", metadata={"tag": "food"}),
                Document(page_content="recipe for cake", metadata={"tag": "food"}),
            ]
        )
        from llm_core.vectorstores.cond import DocCond  # noqa: PLC0415

        results = store.cond_search("recipe", k=5, cond=DocCond("pasta"))
        assert len(results) == 1
        assert "pasta" in results[0].page_content
