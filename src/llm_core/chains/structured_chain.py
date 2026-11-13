"""Generic LangChain structured-output chain with prompt-variable validation.

``StructuredLLMChain[InputT, OutputT]`` wires a Jinja2 prompt template to a
chat model and enforces structured Pydantic output. Input field names are
validated against prompt variables at construction time; the actual chat model
is created lazily on first ``invoke`` / ``ainvoke``. An optional ``on_call``
callback receives the chain's ``name``, the model name and the elapsed seconds
of every call that returns, so a caller can time all its LLM calls in one place.

Example:
    ::

        from llm_core.chains.structured_chain import StructuredLLMChain
        from llm_core.chat.config.openai import ChatOpenAIConfig
        from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
        from pydantic import BaseModel

        class MyInput(BaseModelKwargs):
            recipe_text: str

        class MyOutput(BaseModel):
            name: str
            ingredients: list[str]

        chain = StructuredLLMChain(
            chat_config=ChatOpenAIConfig(),
            prompt_str="Extract recipe from: {{ recipe_text }}",
            input_model=MyInput,
            output_model=MyOutput,
        )
        result: MyOutput = chain.invoke(MyInput(recipe_text="Boil pasta..."))
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from llm_core.chains.exceptions import ExtraPromptVariablesError
from llm_core.chains.exceptions import MissingPromptVariablesError
from llm_core.chat.config.base import ChatConfig
from llm_core.data_models.basemodel_kwargs import BaseModelKwargs


@dataclass
class StructuredLLMChain[InputT: BaseModelKwargs, OutputT: BaseModel]:
    """Reusable LangChain chain with a Jinja2 prompt and structured output.

    The chat model is built lazily on first access to the ``chain`` property
    (triggered by ``invoke`` / ``ainvoke``). Prompt variable validation
    happens eagerly in ``__post_init__`` since it is a cheap, API-free check.

    Attributes:
        chat_config: Provider configuration used to create the chat model.
        prompt_str: Raw Jinja2 template string (system message).
        input_model: Pydantic model whose field names must match prompt variables.
        output_model: Pydantic model enforced via ``with_structured_output``.
        name: Label passed to ``on_call``; empty by default.
        on_call: Called as ``on_call(name, model, elapsed_seconds)`` after a
            call returns the output model. A call that raises, including on a
            wrong output type, reports nothing.
    """

    chat_config: ChatConfig
    prompt_str: str
    input_model: type[InputT]
    output_model: type[OutputT]
    name: str = ""
    on_call: Callable[[str, str, float], None] | None = None
    _chain: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Build prompt template and validate variables against input model."""
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.prompt_str)],
            template_format="jinja2",
        )
        self._validate_prompt_variables()

    def _validate_prompt_variables(self) -> None:
        """Check that input model fields and prompt variables match exactly."""
        input_fields = frozenset(self.input_model.model_fields)
        template_vars = frozenset(self.prompt_template.input_variables)

        missing = input_fields - template_vars
        if missing:
            raise MissingPromptVariablesError(missing)

        extra = template_vars - input_fields
        if extra:
            raise ExtraPromptVariablesError(extra)

    def _build_chain(self) -> Runnable:
        """Create the LCEL chain: prompt | model.with_structured_output."""
        model = self.chat_config.create_chat_model()
        structured_llm = model.with_structured_output(self.output_model)
        return self.prompt_template | structured_llm

    @property
    def chain(self) -> Runnable:
        """LCEL chain, built lazily on first access.

        Returns:
            The composed ``prompt_template | structured_llm`` runnable.
        """
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain

    def _report(self, started: float) -> None:
        """Pass the elapsed time since *started* to ``on_call``, if set."""
        if self.on_call is not None:
            self.on_call(self.name, self.chat_config.model, time.monotonic() - started)

    def invoke(self, chain_input: InputT) -> OutputT:
        """Run the chain synchronously and return the structured output.

        Args:
            chain_input: Input model instance whose fields are rendered into
                the prompt template.

        Returns:
            Validated ``OutputT`` instance.

        Raises:
            TypeError: If the chain returns an unexpected type.
        """
        started = time.monotonic()
        output = self.chain.invoke(chain_input.to_kw())
        if not isinstance(output, self.output_model):
            msg = f"Unexpected output type: {type(output)}"
            raise TypeError(msg)
        self._report(started)
        return output  # type: ignore[return-value]

    async def ainvoke(self, chain_input: InputT) -> OutputT:
        """Run the chain asynchronously and return the structured output.

        Args:
            chain_input: Input model instance whose fields are rendered into
                the prompt template.

        Returns:
            Validated ``OutputT`` instance.

        Raises:
            TypeError: If the chain returns an unexpected type.
        """
        started = time.monotonic()
        output = await self.chain.ainvoke(chain_input.to_kw())
        if not isinstance(output, self.output_model):
            msg = f"Unexpected output type: {type(output)}"
            raise TypeError(msg)
        self._report(started)
        return output  # type: ignore[return-value]
