# Block 5 - StructuredLLMChain

Parent: [01-track-progress.md](01-track-progress.md)
Depends on: Block 2 (chat configs), Block 4 (prompts)

---

## Source reference

Extracted from laife: `src/laife/llm/structured_chain.py`

---

## Files to create

```
src/llm_core/chains/
    __init__.py
    structured_chain.py     # StructuredLLMChain[InputT, OutputT]

src/llm_core/testing/
    __init__.py
    fake_chat_model.py      # FakeChatModel for unit tests (moved to v1 per pitfall #8)
```

---

## Design

### `StructuredLLMChain[InputT, OutputT]`

```python
@dataclass
class StructuredLLMChain[InputT: BaseModelKwargs, OutputT: BaseModel]:
    chat_config: ChatConfig
    prompt_str: str                     # raw Jinja2 template string
    input_model: type[InputT]
    output_model: type[OutputT]
    _chain: Any = field(init=False, default=None)  # built lazily on first access

    def __post_init__(self) -> None:
        # Build prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.prompt_str)], template_format="jinja2"
        )
        # Validate: input fields must match template variables (both directions)
        self._validate_prompt_variables()
        # _chain is NOT built here - use the `chain` property for lazy init

    @property
    def chain(self) -> Runnable:
        """LCEL chain, built on first access."""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain

    def invoke(self, chain_input: InputT) -> OutputT: ...
    async def ainvoke(self, chain_input: InputT) -> OutputT: ...
```

The `lazy` flag is dropped entirely. The property pattern is the standard Python idiom
for deferred initialization - no caller-visible flag needed. `__post_init__` always
validates the prompt variables eagerly (a cheap, API-free operation), while the actual
chat model is created only on the first call to `invoke()` / `ainvoke()` via `self.chain`.

### Prompt variable validation (bidirectional, per pitfall #6)

```python
def _validate_prompt_variables(self) -> None:
    input_fields = frozenset(self.input_model.model_fields)
    template_vars = frozenset(self.prompt_template.input_variables)

    missing = input_fields - template_vars
    if missing:
        raise MissingPromptVariablesError(missing)

    extra = template_vars - input_fields
    if extra:
        raise ExtraPromptVariablesError(extra)
```

### Lazy initialization (per pitfall #5)

All initialization is lazy by default via the `chain` property:
- `__post_init__` validates prompt variables (fast, no network I/O)
- `_build_chain()` is NOT called during construction
- First access to `self.chain` (inside `invoke()` / `ainvoke()`) triggers `_build_chain()`
- `_build_chain()` calls `chat_config.create_chat_model()` once, caches the result in `_chain`
- Useful for config-time chain creation without requiring API connections or credentials

This eliminates the need for a caller-visible `lazy` flag. The behavior is always lazy;
there is no eager-vs-lazy mode to toggle.

### `FakeChatModel` + `FakeChatModelConfig` (per pitfall #8 - moved to v1)

Follows the same config pattern as real providers. `FakeChatModelConfig` is a
`ChatConfig` subclass; `create_chat_model()` returns a `FakeChatModel` that
implements the minimal LangChain `BaseChatModel` interface.

```python
# src/llm_core/testing/fake_chat_model.py

class FakeChatModel(BaseChatModel):
    """Deterministic chat model for unit tests. No API calls."""
    responses: list[BaseMessage]  # pre-loaded replies, cycled in order
    _call_count: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def _llm_type(self) -> str:
        return "fake"


class FakeChatModelConfig(ChatConfig):
    """Config that creates a FakeChatModel for unit tests."""
    responses: list[BaseMessage]
    model: str = "fake"            # satisfies ChatConfig; unused at runtime
    model_provider: str = "fake"   # same

    def create_chat_model(self) -> FakeChatModel:
        return FakeChatModel(responses=self.responses)
```

Usage in tests:

```python
from langchain_core.messages import AIMessage

fake_reply = AIMessage(content='{"name": "Pasta", "ingredients": ["pasta", "water"]}')
chain = StructuredLLMChain(
    chat_config=FakeChatModelConfig(responses=[fake_reply]),
    prompt_str="Extract recipe from: {{ recipe_text }}",
    input_model=RecipeInput,
    output_model=RecipeOutput,
)
```

Key decisions:
- `responses` is a list cycled in order via `_call_count % len(responses)` - no dict keying
  by input hash; keeps tests simpler and deterministic without relying on serialization.
- `FakeChatModel` is a true `BaseChatModel` so it satisfies `with_structured_output()` and
  any other LangChain interface used on chat models - no monkey-patching needed.
- Both classes live in `src/llm_core/testing/fake_chat_model.py` (testing module, not
  under `chat/config/`) to signal that this is not a production provider.

---

## Exceptions

- `MissingPromptVariablesError` - input model has fields not in prompt template
- `ExtraPromptVariablesError` - prompt template has variables not in input model

---

## Tests

```
tests/chains/
    __init__.py
    test_structured_chain.py

tests/testing/
    __init__.py
    test_fake_chat_model.py
```

Test cases for `StructuredLLMChain`:
- Construction with matching input/prompt variables succeeds
- Construction with missing prompt variables raises `MissingPromptVariablesError`
- Construction with extra prompt variables raises `ExtraPromptVariablesError`
- `invoke()` with `FakeChatModelConfig` returns expected output
- `ainvoke()` with `FakeChatModelConfig` returns expected output
- `chain` property is `None` before first invoke (deferred build)
- `chain` property is populated after first invoke (built on demand)
- Second invoke reuses the cached chain (build called only once)

Test cases for `FakeChatModel` / `FakeChatModelConfig`:
- Single response is always returned when `responses` has one element
- Multiple responses are cycled in order (round-robin)
- `FakeChatModelConfig.create_chat_model()` returns a `FakeChatModel` instance
- `FakeChatModel` satisfies `BaseChatModel` (is instance check)

---

## Notes

- `InputT` is bounded by `BaseModelKwargs` so `.to_kw()` can be used to extract
  field values for prompt rendering
- `OutputT` is a plain `BaseModel` - its schema is passed to `with_structured_output()`
- The chain is: `prompt_template | model.with_structured_output(output_model)`
- Both `invoke` and `ainvoke` call `.to_kw()` on the input to get the template variables dict

---

## Notes on `lazy` removal

The original plan listed `lazy: bool = False` as a dataclass field. This is replaced by the
`chain` property. The caller-visible interface does not change - `invoke()` / `ainvoke()` work
the same way. The only observable difference is that `chain` is always `None` at construction
time; there is no way to force eager build (and no reason to, since the property builds
cheaply on first use).

---

## Checklist

- [ ] Read laife's structured_chain.py for exact implementation
- [ ] Create StructuredLLMChain with bidirectional validation
- [ ] Add `chain` property for lazy initialization (no `lazy` flag)
- [ ] Create `FakeChatModel(BaseChatModel)` and `FakeChatModelConfig(ChatConfig)` in testing module
- [ ] Create MissingPromptVariablesError, ExtraPromptVariablesError
- [ ] Write tests using FakeChatModel
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
