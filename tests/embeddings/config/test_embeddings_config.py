"""Tests for EmbeddingsConfig base class."""

from llm_core.embeddings.config.base import EmbeddingsConfig


class ConcreteEmbeddingsConfig(EmbeddingsConfig):
    """Minimal concrete subclass for testing the abstract base."""

    model: str = "test-model"
    provider: str = "test-provider"


def test_to_kw_contains_required_fields() -> None:
    """to_kw includes model and provider."""
    cfg = ConcreteEmbeddingsConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["model"] == "test-model"
    assert kw["provider"] == "test-provider"


def test_create_embeddings_is_callable() -> None:
    """create_embeddings exists and is callable (not testing actual calls)."""
    cfg = ConcreteEmbeddingsConfig()
    assert callable(cfg.create_embeddings)
