"""Tests for AzureOpenAIChatConfig."""

from pydantic import SecretStr

from llm_core.chat.config.azure_openai import AzureOpenAIChatConfig


def test_defaults() -> None:
    """Verify default model, provider, and api_version."""
    cfg = AzureOpenAIChatConfig()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.model_provider == "azure_openai"
    assert cfg.api_version == "2024-02-01"


def test_to_kw_provider() -> None:
    """to_kw includes model_provider and api_version."""
    cfg = AzureOpenAIChatConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["model_provider"] == "azure_openai"
    assert kw["api_version"] == "2024-02-01"


def test_api_key_secret_str() -> None:
    """api_key is stored as SecretStr."""
    cfg = AzureOpenAIChatConfig(api_key=SecretStr("az-key"))
    assert isinstance(cfg.api_key, SecretStr)


def test_api_key_masked_in_str() -> None:
    """api_key secret value is redacted in string representation."""
    cfg = AzureOpenAIChatConfig(api_key=SecretStr("az-secret"))
    assert "az-secret" not in str(cfg)
    assert "az-secret" not in repr(cfg)


def test_azure_endpoint_excluded_when_none() -> None:
    """None azure_endpoint is excluded from to_kw(exclude_none=True)."""
    cfg = AzureOpenAIChatConfig(azure_endpoint=None)
    kw = cfg.to_kw(exclude_none=True)
    assert "azure_endpoint" not in kw


def test_azure_endpoint_included_when_set() -> None:
    """Set azure_endpoint appears in to_kw output."""
    cfg = AzureOpenAIChatConfig(azure_endpoint="https://my.azure.com")
    kw = cfg.to_kw(exclude_none=True)
    assert kw["azure_endpoint"] == "https://my.azure.com"
