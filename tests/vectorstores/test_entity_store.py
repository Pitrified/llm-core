"""Tests for EntityStore facade."""

from typing import TYPE_CHECKING
from typing import Self
import uuid

from langchain_core.documents import Document
import pytest

try:
    import chromadb  # noqa: F401

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

if TYPE_CHECKING:
    from llm_core.vectorstores.entity_store import EntityStore

pytestmark = pytest.mark.skipif(not HAS_CHROMA, reason="chromadb not installed")


class SampleEntity:
    """Dummy Vectorable entity for testing."""

    def __init__(self, text: str, entity_type: str = "sample") -> None:
        """Store text and entity type."""
        self.text = text
        self.entity_type = entity_type

    def to_document(self) -> Document:  # noqa: D102
        return Document(
            page_content=self.text,
            metadata={"entity_type": self.entity_type},
        )

    @classmethod
    def from_document(cls, doc: Document) -> Self:  # noqa: D102
        return cls(
            text=doc.page_content,
            entity_type=doc.metadata.get("entity_type", "sample"),
        )


def _make_store() -> "EntityStore":
    from llm_core.vectorstores.config.chroma import ChromaConfig  # noqa: PLC0415
    from llm_core.vectorstores.entity_store import EntityStore  # noqa: PLC0415

    config = ChromaConfig(collection_name=f"test_{uuid.uuid4().hex[:8]}")
    return EntityStore(config=config)


class TestEntityStoreSave:
    """Tests for save / save_many."""

    def test_save_single_entity(self) -> None:
        """Saving a single entity succeeds and is searchable."""
        store = _make_store()
        entity = SampleEntity("hello world")
        store.save(entity)

        docs = store.search("hello", k=1)
        assert len(docs) == 1
        assert docs[0].page_content == "hello world"

    def test_save_many_entities(self) -> None:
        """Saving multiple entities succeeds."""
        store = _make_store()
        entities = [SampleEntity(f"entity {i}") for i in range(3)]
        store.save(entities)

        docs = store.search("entity", k=5)
        assert len(docs) == 3


class TestEntityStoreSearch:
    """Tests for search overloads."""

    def test_search_returns_documents(self) -> None:
        """search() without entity_type returns list[Document]."""
        store = _make_store()
        store.save(SampleEntity("test content"))

        docs = store.search("test", k=1)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_search_with_entity_type_returns_typed(self) -> None:
        """search() with entity_type returns list of typed entities."""
        store = _make_store()
        store.save(SampleEntity("typed content"))

        results = store.search("typed", entity_type=SampleEntity, k=1)
        assert len(results) == 1
        assert isinstance(results[0], SampleEntity)
        assert results[0].text == "typed content"

    def test_search_no_entity_and_cond(self) -> None:
        """search() without entity_type but with cond filters correctly."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415

        store = _make_store()
        store.save(SampleEntity("alpha", entity_type="type_a"))
        store.save(SampleEntity("beta", entity_type="type_b"))

        results = store.search(
            "alpha",
            k=5,
            cond=CompCond("entity_type", CompOp.EQ, "type_a"),
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].page_content == "alpha"

    def test_search_with_entity_and_cond(self) -> None:
        """search() with entity_type and cond filters and deserializes."""
        from llm_core.vectorstores.cond import CompCond  # noqa: PLC0415
        from llm_core.vectorstores.cond import CompOp  # noqa: PLC0415

        store = _make_store()
        store.save(SampleEntity("alpha", entity_type="type_a"))
        store.save(SampleEntity("beta", entity_type="type_b"))

        results = store.search(
            "alpha",
            entity_type=SampleEntity,
            k=5,
            cond=CompCond("entity_type", CompOp.EQ, "type_a"),
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SampleEntity)
        assert results[0].text == "alpha"
