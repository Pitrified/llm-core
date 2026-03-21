"""Tests for OpenAIEmbeddingsConfig."""

from pydantic import SecretStr

from llm_core.embeddings.config.openai import OpenAIEmbeddingsConfig


def test_defaults() -> None:
    """Verify default model and provider."""
    cfg = OpenAIEmbeddingsConfig()
    assert cfg.model == "text-embedding-3-small"
    assert cfg.provider == "openai"


def test_to_kw_provider() -> None:
    """to_kw includes provider and model."""
    cfg = OpenAIEmbeddingsConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["provider"] == "openai"
    assert kw["model"] == "text-embedding-3-small"


def test_api_key_secret_str() -> None:
    """api_key is stored as SecretStr."""
    cfg = OpenAIEmbeddingsConfig(api_key=SecretStr("sk-test"))
    assert isinstance(cfg.api_key, SecretStr)


def test_api_key_masked_in_str() -> None:
    """api_key secret value is redacted in string representation."""
    cfg = OpenAIEmbeddingsConfig(api_key=SecretStr("sk-secret"))
    assert "sk-secret" not in str(cfg)
    assert "sk-secret" not in repr(cfg)


def test_api_key_excluded_when_none() -> None:
    """None api_key is excluded from to_kw(exclude_none=True)."""
    cfg = OpenAIEmbeddingsConfig(api_key=None)
    kw = cfg.to_kw(exclude_none=True)
    assert "api_key" not in kw
