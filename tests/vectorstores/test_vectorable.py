"""Tests for the Vectorable protocol."""

from typing import Self

from langchain_core.documents import Document

from llm_core.vectorstores.vectorable import Vectorable


class DummyEntity:
    """Dummy entity that satisfies Vectorable."""

    def __init__(self, text: str) -> None:
        """Store text."""
        self.text = text

    def to_document(self) -> Document:  # noqa: D102
        return Document(
            page_content=self.text,
            metadata={"entity_type": "dummy"},
        )

    @classmethod
    def from_document(cls, doc: Document) -> Self:  # noqa: D102
        return cls(text=doc.page_content)


class NotVectorable:
    """Class that does NOT satisfy Vectorable."""

    def some_method(self) -> None:  # noqa: D102
        ...


class TestVectorableProtocol:
    """Tests for Vectorable protocol conformance."""

    def test_dummy_entity_satisfies_protocol(self) -> None:
        """A class with to_document and from_document satisfies Vectorable."""
        entity = DummyEntity("hello")
        assert isinstance(entity, Vectorable)

    def test_non_conforming_class_fails(self) -> None:
        """A class without the required methods does not satisfy Vectorable."""
        obj = NotVectorable()
        assert not isinstance(obj, Vectorable)

    def test_to_document_round_trip(self) -> None:
        """to_document and from_document produce a valid round-trip."""
        entity = DummyEntity("test content")
        doc = entity.to_document()
        assert doc.page_content == "test content"
        assert doc.metadata["entity_type"] == "dummy"

        restored = DummyEntity.from_document(doc)
        assert restored.text == "test content"
