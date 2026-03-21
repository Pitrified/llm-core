"""Deterministic chat model and config for unit tests.

Provides ``FakeChatModel`` (a real ``BaseChatModel`` subclass) and
``FakeChatModelConfig`` (a ``ChatConfig`` subclass) so that consumers can
build and exercise ``StructuredLLMChain`` instances without making any API
calls.

Example:
    ::

        from langchain_core.messages import AIMessage

        fake_reply = AIMessage(content='{"name": "Pasta"}')
        config = FakeChatModelConfig(responses=[fake_reply])
        model = config.create_chat_model()  # -> FakeChatModel
"""

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from pydantic import PrivateAttr

from llm_core.chat.config.base import ChatConfig


class FakeChatModel(BaseChatModel):
    """Deterministic chat model for unit tests. No API calls.

    Cycles through *responses* in round-robin order. Each call to
    ``_generate`` returns the next response in the list.

    Supports ``with_structured_output`` by parsing the JSON content of each
    response and validating it against the given Pydantic schema.

    Attributes:
        responses: Pre-loaded replies cycled in order.
    """

    responses: list[BaseMessage]

    _call_count: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> ChatResult:
        reply = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def _llm_type(self) -> str:
        return "fake"

    def with_structured_output(
        self,
        schema: type[BaseModel] | dict[str, Any],
        *,
        include_raw: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> Runnable:
        """Return a runnable that parses model output as *schema*.

        Args:
            schema: Pydantic model class or JSON schema dict.
            include_raw: Ignored for this fake implementation.
            kwargs: Ignored.

        Returns:
            A runnable that invokes this model and parses the JSON content
            into a *schema* instance.
        """

        def _parse(input_val: Any) -> Any:  # noqa: ANN401
            result = self.invoke(input_val)
            content = result.content
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(json.loads(str(content)))
            return json.loads(str(content))

        return RunnableLambda(_parse)


class FakeChatModelConfig(ChatConfig):
    """Chat config that creates a ``FakeChatModel`` for unit tests.

    Attributes:
        responses: Pre-loaded replies forwarded to the ``FakeChatModel``.
        model: Satisfies ``ChatConfig``; unused at runtime.
        model_provider: Satisfies ``ChatConfig``; unused at runtime.
    """

    responses: list[BaseMessage]
    model: str = "fake"
    model_provider: str = "fake"

    def create_chat_model(self) -> FakeChatModel:
        """Return a ``FakeChatModel`` with the configured responses.

        Returns:
            A deterministic chat model that cycles through *responses*.
        """
        return FakeChatModel(responses=self.responses)
