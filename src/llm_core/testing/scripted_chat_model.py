"""Chat model that answers structured-output calls by schema, with no API calls.

``FakeChatModel`` cycles one list of replies whatever the schema; this model
looks the requested output model up in a mapping instead, so several chains
built on one config each get an answer of their own type. It is meant for
running a whole application without a provider, not only for unit tests.

Example:
    ::

        from llm_core.testing.scripted_chat_model import Answers
        from llm_core.testing.scripted_chat_model import ScriptedChatConfig

        class GameScriptConfig(ScriptedChatConfig):
            def build_answers(self) -> Answers:
                return {Verdict: lambda: Verdict(success=True)}

        chain = StructuredLLMChain(chat_config=GameScriptConfig(), ...)
"""

from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from llm_core.chat.config.base import ChatConfig

type Answers = dict[type[BaseModel], Callable[[], BaseModel]]
"""Output model to a callable returning an instance of it."""


class UnscriptedSchemaError(Exception):
    """A chain asked the scripted model for an output model it has no answer for."""

    def __init__(self, schema: object) -> None:
        """Name the schema, so a new chain shows up as this failure."""
        super().__init__(f"No scripted answer for output model {schema!r}")


class UnscriptedCallError(Exception):
    """The scripted model was called for free text; only structured output is."""

    def __init__(self) -> None:
        """Explain which call is supported."""
        super().__init__(
            "The scripted model only answers with_structured_output calls."
        )


class ScriptedChatModel(BaseChatModel):
    """Chat model that answers ``with_structured_output(schema)`` from *answers*.

    The prompt is ignored. A plain ``invoke`` raises ``UnscriptedCallError``.

    Attributes:
        answers: Output model to a callable returning an instance of it.
    """

    answers: Answers

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> ChatResult:
        raise UnscriptedCallError

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def with_structured_output(
        self,
        schema: type[BaseModel] | dict[str, Any],
        *,
        include_raw: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> Runnable:
        """Return a runnable that ignores the prompt and answers for *schema*.

        Args:
            schema: The output model the chain asks for.
            include_raw: Ignored.
            kwargs: Ignored.

        Returns:
            A runnable returning ``answer(schema)``.
        """
        return RunnableLambda(lambda _prompt: self.answer(schema))

    def answer(self, schema: object) -> BaseModel:
        """Return the scripted instance of *schema*.

        Raises:
            UnscriptedSchemaError: If *schema* has no entry in ``answers``.
        """
        make = self.answers.get(schema)  # type: ignore[arg-type]
        if make is None:
            raise UnscriptedSchemaError(schema)
        return make()


class ScriptedChatConfig(ChatConfig):
    """Chat config that builds a :class:`ScriptedChatModel`.

    Subclass it and override :meth:`build_answers`. Each chain builds its own
    model from the config, and each model gets the answers of its own
    ``build_answers`` call, so state kept in the callables (a cycle through a
    script) is per chain, not shared through the config.

    Attributes:
        model: Model name reported by ``StructuredLLMChain.on_call``.
        model_provider: Satisfies ``ChatConfig``; unused.
    """

    model: str = "scripted"
    model_provider: str = "scripted"

    def build_answers(self) -> Answers:
        """Return the answers of one model; none by default."""
        return {}

    def create_chat_model(self) -> ScriptedChatModel:
        """Return a scripted model with freshly built answers."""
        return ScriptedChatModel(answers=self.build_answers())
