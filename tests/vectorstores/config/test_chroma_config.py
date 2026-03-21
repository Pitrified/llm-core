"""Tests for ChromaConfig."""

from llm_core.vectorstores.config.chroma import ChromaConfig


class TestChromaConfig:
    """Tests for ChromaConfig."""

    def test_default_fields(self) -> None:
        """Default values are sensible."""
        config = ChromaConfig()
        assert config.collection_name == "default"
        assert config.persist_directory is None
        assert config.host is None
        assert config.port is None
        assert config.ssl is False
        assert config.embeddings_config is None

    def test_custom_fields(self) -> None:
        """Custom values are applied."""
        config = ChromaConfig(
            collection_name="my_collection",
            persist_directory="/tmp/chroma",  # noqa: S108
        )
        assert config.collection_name == "my_collection"
        assert config.persist_directory == "/tmp/chroma"  # noqa: S108

    def test_to_kw_output(self) -> None:
        """to_kw() returns the expected dict."""
        config = ChromaConfig(
            collection_name="test",
            persist_directory="/data/chroma",
        )
        kw = config.to_kw(exclude_none=True)
        assert kw["collection_name"] == "test"
        assert kw["persist_directory"] == "/data/chroma"
        assert "host" not in kw
        assert "port" not in kw
