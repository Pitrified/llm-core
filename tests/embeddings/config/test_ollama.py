"""Tests for OllamaEmbeddingsConfig."""

from llm_core.embeddings.config.ollama import OllamaEmbeddingsConfig


def test_defaults() -> None:
    """Verify default model and provider."""
    cfg = OllamaEmbeddingsConfig()
    assert cfg.model == "nomic-embed-text"
    assert cfg.provider == "ollama"


def test_to_kw_provider() -> None:
    """to_kw includes provider and model."""
    cfg = OllamaEmbeddingsConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["provider"] == "ollama"
    assert kw["model"] == "nomic-embed-text"


def test_base_url_excluded_when_none() -> None:
    """None base_url is excluded from to_kw(exclude_none=True)."""
    cfg = OllamaEmbeddingsConfig(base_url=None)
    kw = cfg.to_kw(exclude_none=True)
    assert "base_url" not in kw


def test_base_url_included_when_set() -> None:
    """Set base_url appears in to_kw output."""
    cfg = OllamaEmbeddingsConfig(base_url="http://my-server:11434")
    kw = cfg.to_kw(exclude_none=True)
    assert kw["base_url"] == "http://my-server:11434"
