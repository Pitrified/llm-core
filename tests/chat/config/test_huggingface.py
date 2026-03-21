"""Tests for HuggingFaceChatConfig."""

from pydantic import SecretStr

from llm_core.chat.config.huggingface import HuggingFaceChatConfig


def test_defaults() -> None:
    """Verify default model, provider, and max_tokens."""
    cfg = HuggingFaceChatConfig()
    assert cfg.model == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert cfg.model_provider == "huggingface"
    assert cfg.max_tokens == 1024


def test_to_kw_provider() -> None:
    """to_kw includes model_provider."""
    cfg = HuggingFaceChatConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["model_provider"] == "huggingface"


def test_api_key_secret_str() -> None:
    """api_key is stored as SecretStr."""
    cfg = HuggingFaceChatConfig(api_key=SecretStr("hf-token"))
    assert isinstance(cfg.api_key, SecretStr)


def test_api_key_masked_in_str() -> None:
    """api_key secret value is redacted in string representation."""
    cfg = HuggingFaceChatConfig(api_key=SecretStr("hf-secret"))
    assert "hf-secret" not in str(cfg)
    assert "hf-secret" not in repr(cfg)


def test_api_key_excluded_when_none() -> None:
    """None api_key is excluded from to_kw(exclude_none=True)."""
    cfg = HuggingFaceChatConfig(api_key=None)
    kw = cfg.to_kw(exclude_none=True)
    assert "api_key" not in kw
