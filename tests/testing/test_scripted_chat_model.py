"""Tests for ScriptedChatModel and ScriptedChatConfig."""

from itertools import count

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import pytest

from llm_core.chains.structured_chain import StructuredLLMChain
from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
from llm_core.testing.scripted_chat_model import Answers
from llm_core.testing.scripted_chat_model import ScriptedChatConfig
from llm_core.testing.scripted_chat_model import ScriptedChatModel
from llm_core.testing.scripted_chat_model import UnscriptedCallError
from llm_core.testing.scripted_chat_model import UnscriptedSchemaError


class Verdict(BaseModel):
    """Sample output model."""

    success: bool


class Step(BaseModel):
    """Sample output model whose answers change per call."""

    n: int


class Question(BaseModelKwargs):
    """Sample input model."""

    text: str


class CountingConfig(ScriptedChatConfig):
    """Answers Verdict with a fixed value and Step with a counter per model."""

    def build_answers(self) -> Answers:
        """Return a fresh counter for each model."""
        steps = count()
        return {
            Verdict: lambda: Verdict(success=True),
            Step: lambda: Step(n=next(steps)),
        }


def test_config_defaults() -> None:
    """Model and provider are 'scripted'; no answers by default."""
    config = ScriptedChatConfig()
    assert config.model == "scripted"
    assert config.model_provider == "scripted"
    assert config.create_chat_model().answers == {}


def test_answers_by_schema() -> None:
    """Each schema gets its own answer."""
    model = CountingConfig().create_chat_model()
    assert model.answer(Verdict) == Verdict(success=True)
    assert model.answer(Step) == Step(n=0)
    assert model.answer(Step) == Step(n=1)


def test_each_model_gets_fresh_answers() -> None:
    """Two models built from one config do not share the counter."""
    config = CountingConfig()
    first = config.create_chat_model()
    second = config.create_chat_model()
    first.answer(Step)
    assert second.answer(Step) == Step(n=0)


def test_unknown_schema_raises() -> None:
    """An output model with no answer fails by name."""
    model = CountingConfig().create_chat_model()
    with pytest.raises(UnscriptedSchemaError, match="Question"):
        model.answer(Question)


def test_free_text_call_raises() -> None:
    """A plain invoke is not scripted."""
    model = ScriptedChatModel(answers={})
    with pytest.raises(UnscriptedCallError):
        model.invoke([HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_answers_through_structured_chain() -> None:
    """A StructuredLLMChain on the config returns the scripted answer."""
    chain = StructuredLLMChain(
        chat_config=CountingConfig(),
        prompt_str="{{ text }}",
        input_model=Question,
        output_model=Verdict,
    )
    assert await chain.ainvoke(Question(text="ok?")) == Verdict(success=True)
