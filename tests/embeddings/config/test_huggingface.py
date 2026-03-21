"""Tests for HuggingFaceEmbeddingsConfig."""

from llm_core.embeddings.config.huggingface import HuggingFaceEmbeddingsConfig


def test_defaults() -> None:
    """Verify default model and provider."""
    cfg = HuggingFaceEmbeddingsConfig()
    assert cfg.model == "sentence-transformers/all-mpnet-base-v2"
    assert cfg.provider == "huggingface"


def test_to_kw_provider() -> None:
    """to_kw includes provider and model."""
    cfg = HuggingFaceEmbeddingsConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["provider"] == "huggingface"
    assert kw["model"] == "sentence-transformers/all-mpnet-base-v2"


def test_custom_model() -> None:
    """Custom model name is stored correctly."""
    cfg = HuggingFaceEmbeddingsConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    assert cfg.model == "sentence-transformers/all-MiniLM-L6-v2"
