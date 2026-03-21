"""Tests for ChatOpenAIConfig."""

from pydantic import SecretStr

from llm_core.chat.config.openai import ChatOpenAIConfig


def test_defaults() -> None:
    """Verify default model and provider."""
    cfg = ChatOpenAIConfig()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.model_provider == "openai"


def test_to_kw_provider() -> None:
    """to_kw includes model_provider and model."""
    cfg = ChatOpenAIConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["model_provider"] == "openai"
    assert kw["model"] == "gpt-4o-mini"


def test_api_key_secret_str() -> None:
    """api_key is stored as SecretStr."""
    cfg = ChatOpenAIConfig(api_key=SecretStr("sk-test"))
    assert isinstance(cfg.api_key, SecretStr)


def test_api_key_masked_in_str() -> None:
    """api_key secret value is redacted in string representation."""
    cfg = ChatOpenAIConfig(api_key=SecretStr("sk-test"))
    assert "sk-test" not in str(cfg)
    assert "sk-test" not in repr(cfg)


def test_api_key_excluded_when_none() -> None:
    """None api_key is excluded from to_kw(exclude_none=True)."""
    cfg = ChatOpenAIConfig(api_key=None)
    kw = cfg.to_kw(exclude_none=True)
    assert "api_key" not in kw
