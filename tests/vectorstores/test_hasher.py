"""Tests for the document ID hasher."""

from llm_core.vectorstores.hasher import document_id


class TestDocumentId:
    """Tests for document_id function."""

    def test_deterministic(self) -> None:
        """Same content and metadata produce the same ID."""
        id1 = document_id("hello", {"key": "value"})
        id2 = document_id("hello", {"key": "value"})
        assert id1 == id2

    def test_different_content_different_id(self) -> None:
        """Different content produces different IDs."""
        id1 = document_id("hello", {"key": "value"})
        id2 = document_id("world", {"key": "value"})
        assert id1 != id2

    def test_different_metadata_different_id(self) -> None:
        """Different metadata produces different IDs."""
        id1 = document_id("hello", {"key": "value1"})
        id2 = document_id("hello", {"key": "value2"})
        assert id1 != id2

    def test_empty_inputs(self) -> None:
        """Empty content and metadata produce a valid hex string."""
        result = document_id("", {})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_metadata_order_irrelevant(self) -> None:
        """Metadata key order does not affect the hash (sorted internally)."""
        id1 = document_id("x", {"a": 1, "b": 2})
        id2 = document_id("x", {"b": 2, "a": 1})
        assert id1 == id2

    def test_returns_64_char_hex(self) -> None:
        """Output is always a 64-character hex string (SHA-256)."""
        result = document_id("some content", {"entity_type": "test"})
        assert len(result) == 64
