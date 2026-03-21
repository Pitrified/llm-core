"""Tests for ChatConfig base class."""

from llm_core.chat.config.base import ChatConfig


class ConcreteChatConfig(ChatConfig):
    """Minimal concrete subclass for testing the abstract base."""

    model: str = "test-model"
    model_provider: str = "test-provider"


def test_default_temperature() -> None:
    """Verify temperature defaults to 0.2."""
    cfg = ConcreteChatConfig()
    assert cfg.temperature == 0.2


def test_to_kw_contains_required_fields() -> None:
    """to_kw includes model, model_provider, and temperature."""
    cfg = ConcreteChatConfig()
    kw = cfg.to_kw(exclude_none=True)
    assert kw["model"] == "test-model"
    assert kw["model_provider"] == "test-provider"
    assert kw["temperature"] == 0.2


def test_to_kw_custom_temperature() -> None:
    """Custom temperature is forwarded through to_kw."""
    cfg = ConcreteChatConfig(temperature=0.9)
    assert cfg.to_kw(exclude_none=True)["temperature"] == 0.9


def test_create_chat_model_is_callable() -> None:
    """create_chat_model exists and is callable (not testing LLM calls)."""
    cfg = ConcreteChatConfig()
    assert callable(cfg.create_chat_model)
