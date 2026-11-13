"""Tests for StructuredLLMChain."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import pytest

from llm_core.chains.exceptions import ExtraPromptVariablesError
from llm_core.chains.exceptions import MissingPromptVariablesError
from llm_core.chains.structured_chain import StructuredLLMChain
from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
from llm_core.testing.fake_chat_model import FakeChatModelConfig

# -- fixtures ----------------------------------------------------------------


class RecipeInput(BaseModelKwargs):
    """Sample input model for tests."""

    recipe_text: str


class RecipeOutput(BaseModel):
    """Sample output model for tests."""

    name: str
    ingredients: list[str]


PROMPT = "Extract recipe from: {{ recipe_text }}"


def _make_fake_config(
    content: str = '{"name":"Pasta","ingredients":["pasta","water"]}',
) -> FakeChatModelConfig:
    return FakeChatModelConfig(responses=[AIMessage(content=content)])


# -- construction tests ------------------------------------------------------


class TestConstruction:
    """Construction and prompt variable validation."""

    def test_matching_variables_succeeds(self) -> None:
        """Construction with matching input/prompt variables succeeds."""
        chain = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        assert chain.input_model is RecipeInput
        assert chain.output_model is RecipeOutput

    def test_missing_prompt_variables_raises(self) -> None:
        """Input model field not in prompt raises MissingPromptVariablesError."""

        class TwoFieldInput(BaseModelKwargs):
            recipe_text: str
            author: str

        with pytest.raises(MissingPromptVariablesError, match="author"):
            StructuredLLMChain(
                chat_config=_make_fake_config(),
                prompt_str=PROMPT,
                input_model=TwoFieldInput,
                output_model=RecipeOutput,
            )

    def test_extra_prompt_variables_raises(self) -> None:
        """Prompt variable not in input model raises ExtraPromptVariablesError."""

        class EmptyInput(BaseModelKwargs):
            pass

        with pytest.raises(ExtraPromptVariablesError, match="recipe_text"):
            StructuredLLMChain(
                chat_config=_make_fake_config(),
                prompt_str=PROMPT,
                input_model=EmptyInput,
                output_model=RecipeOutput,
            )


# -- lazy initialization tests ----------------------------------------------


class TestLazyInit:
    """Chain property is built lazily on first access."""

    def test_chain_is_none_before_access(self) -> None:
        """_chain is None immediately after construction."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        assert sc._chain is None  # noqa: SLF001

    def test_chain_property_builds_on_first_access(self) -> None:
        """Accessing .chain triggers _build_chain and caches the result."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        chain = sc.chain
        assert chain is not None
        assert sc._chain is chain  # noqa: SLF001

    def test_chain_property_reuses_cached(self) -> None:
        """Second access to .chain returns the same object (no rebuild)."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        first = sc.chain
        second = sc.chain
        assert first is second


# -- invoke / ainvoke tests --------------------------------------------------


class TestInvoke:
    """Synchronous and asynchronous invocation."""

    def test_invoke_returns_output(self) -> None:
        """invoke() with FakeChatModelConfig returns expected RecipeOutput."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        result = sc.invoke(RecipeInput(recipe_text="Boil pasta in water"))
        assert isinstance(result, RecipeOutput)
        assert result.name == "Pasta"
        assert result.ingredients == ["pasta", "water"]

    @pytest.mark.asyncio
    async def test_ainvoke_returns_output(self) -> None:
        """ainvoke() with FakeChatModelConfig returns expected RecipeOutput."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        result = await sc.ainvoke(RecipeInput(recipe_text="Boil pasta in water"))
        assert isinstance(result, RecipeOutput)
        assert result.name == "Pasta"
        assert result.ingredients == ["pasta", "water"]

    def test_invoke_raises_on_wrong_output_type(self) -> None:
        """invoke() raises TypeError when the model returns something else."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        sc._chain = RunnableLambda(lambda _: "not a recipe")  # noqa: SLF001
        with pytest.raises(TypeError, match="Unexpected output type"):
            sc.invoke(RecipeInput(recipe_text="Boil pasta in water"))

    @pytest.mark.asyncio
    async def test_ainvoke_raises_on_wrong_output_type(self) -> None:
        """ainvoke() raises TypeError when the model returns something else."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        sc._chain = RunnableLambda(lambda _: "not a recipe")  # noqa: SLF001
        with pytest.raises(TypeError, match="Unexpected output type"):
            await sc.ainvoke(RecipeInput(recipe_text="Boil pasta in water"))


# -- on_call tests -----------------------------------------------------------


class _Calls:
    """Records the arguments of each on_call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, name: str, model: str, elapsed: float) -> None:
        self.calls.append((name, model, elapsed))


def _timed_chain(calls: _Calls) -> StructuredLLMChain[RecipeInput, RecipeOutput]:
    return StructuredLLMChain(
        chat_config=_make_fake_config(),
        prompt_str=PROMPT,
        input_model=RecipeInput,
        output_model=RecipeOutput,
        name="recipe",
        on_call=calls,
    )


class TestOnCall:
    """on_call receives name, model and elapsed seconds of each returned call."""

    def test_invoke_reports(self) -> None:
        """invoke() calls on_call once with the chain name and model."""
        calls = _Calls()
        _timed_chain(calls).invoke(RecipeInput(recipe_text="Boil pasta in water"))
        assert len(calls.calls) == 1
        name, model, elapsed = calls.calls[0]
        assert (name, model) == ("recipe", "fake")
        assert elapsed >= 0

    @pytest.mark.asyncio
    async def test_ainvoke_reports(self) -> None:
        """ainvoke() calls on_call once with the chain name and model."""
        calls = _Calls()
        await _timed_chain(calls).ainvoke(
            RecipeInput(recipe_text="Boil pasta in water")
        )
        assert len(calls.calls) == 1
        name, model, elapsed = calls.calls[0]
        assert (name, model) == ("recipe", "fake")
        assert elapsed >= 0

    def test_wrong_output_type_reports_nothing(self) -> None:
        """A call that raises on the output type does not call on_call."""
        calls = _Calls()
        sc = _timed_chain(calls)
        sc._chain = RunnableLambda(lambda _: "not a recipe")  # noqa: SLF001
        with pytest.raises(TypeError):
            sc.invoke(RecipeInput(recipe_text="Boil pasta in water"))
        assert calls.calls == []

    @pytest.mark.asyncio
    async def test_failing_call_reports_nothing(self) -> None:
        """A call whose model raises does not call on_call."""
        calls = _Calls()
        sc = _timed_chain(calls)

        def _fail(_: object) -> None:
            raise RuntimeError

        sc._chain = RunnableLambda(_fail)  # noqa: SLF001
        with pytest.raises(RuntimeError):
            await sc.ainvoke(RecipeInput(recipe_text="Boil pasta in water"))
        assert calls.calls == []

    def test_no_callback(self) -> None:
        """Without on_call the chain runs as before."""
        sc = StructuredLLMChain(
            chat_config=_make_fake_config(),
            prompt_str=PROMPT,
            input_model=RecipeInput,
            output_model=RecipeOutput,
        )
        assert sc.on_call is None
        assert sc.name == ""
        assert sc.invoke(RecipeInput(recipe_text="x")).name == "Pasta"
