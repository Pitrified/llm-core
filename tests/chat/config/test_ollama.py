"""Tests for OllamaChatConfig."""

from llm_core.chat.config.ollama import OllamaChatConfig


def test_defaults() -> None:
    """Verify default model, provider, and max_tokens."""
    cfg = OllamaChatConfig()
    assert cfg.model == "llama3.2"
    assert cfg.model_provider == "ollama"
    assert cfg.max_tokens == 1024


def test_to_kw_provider() -> None:
    """to_kw includes model_provider and model."""
    cfg = OllamaChatConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["model_provider"] == "ollama"
    assert kw["model"] == "llama3.2"


def test_base_url_excluded_when_none() -> None:
    """None base_url is excluded from to_kw(exclude_none=True)."""
    cfg = OllamaChatConfig(base_url=None)
    kw = cfg.to_kw(exclude_none=True)
    assert "base_url" not in kw


def test_base_url_included_when_set() -> None:
    """Set base_url appears in to_kw output."""
    cfg = OllamaChatConfig(base_url="http://my-server:11434")
    kw = cfg.to_kw(exclude_none=True)
    assert kw["base_url"] == "http://my-server:11434"


def test_custom_model() -> None:
    """Custom model name is stored correctly."""
    cfg = OllamaChatConfig(model="mistral")
    assert cfg.model == "mistral"
