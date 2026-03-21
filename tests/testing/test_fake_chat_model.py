"""Tests for FakeChatModel and FakeChatModelConfig."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from llm_core.testing.fake_chat_model import FakeChatModel
from llm_core.testing.fake_chat_model import FakeChatModelConfig


class TestFakeChatModelConfig:
    """Tests for FakeChatModelConfig."""

    def test_create_chat_model_returns_fake(self) -> None:
        """create_chat_model() returns a FakeChatModel instance."""
        reply = AIMessage(content="hello")
        config = FakeChatModelConfig(responses=[reply])
        model = config.create_chat_model()
        assert isinstance(model, FakeChatModel)

    def test_default_fields(self) -> None:
        """Default model and model_provider are 'fake'."""
        config = FakeChatModelConfig(responses=[AIMessage(content="x")])
        assert config.model == "fake"
        assert config.model_provider == "fake"


class TestFakeChatModel:
    """Tests for FakeChatModel."""

    def test_is_base_chat_model(self) -> None:
        """FakeChatModel satisfies BaseChatModel isinstance check."""
        model = FakeChatModel(responses=[AIMessage(content="hi")])
        assert isinstance(model, BaseChatModel)

    def test_llm_type(self) -> None:
        """_llm_type returns 'fake'."""
        model = FakeChatModel(responses=[AIMessage(content="hi")])
        assert model._llm_type == "fake"  # noqa: SLF001

    def test_single_response_always_returned(self) -> None:
        """With one response, every invoke returns the same message."""
        reply = AIMessage(content="always this")
        model = FakeChatModel(responses=[reply])
        for _ in range(3):
            result = model.invoke([HumanMessage(content="anything")])
            assert result.content == "always this"

    def test_multiple_responses_cycled(self) -> None:
        """With multiple responses, they are cycled in round-robin order."""
        r1 = AIMessage(content="first")
        r2 = AIMessage(content="second")
        r3 = AIMessage(content="third")
        model = FakeChatModel(responses=[r1, r2, r3])

        results = [model.invoke([HumanMessage(content="q")]).content for _ in range(6)]
        assert results == ["first", "second", "third", "first", "second", "third"]
