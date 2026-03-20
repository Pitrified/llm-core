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
    lazy: bool = False                  # defer model creation to first invoke

    def __post_init__(self) -> None:
        # Build prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.prompt_str)], template_format="jinja2"
        )
        # Validate: input fields must match template variables (both directions)
        self._validate_prompt_variables()
        # Build chain (unless lazy)
        if not self.lazy:
            self._build_chain()

    def invoke(self, chain_input: InputT) -> OutputT: ...
    async def ainvoke(self, chain_input: InputT) -> OutputT: ...
```

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

When `lazy=True`:
- `__post_init__` validates prompt variables but does NOT call `create_chat_model()`
- First call to `invoke()` / `ainvoke()` calls `_build_chain()` if not yet built
- Useful for config-time chain creation without requiring API connections

### `FakeChatModel` (per pitfall #8 - moved to v1)

```python
class FakeChatModel:
    """Deterministic chat model for unit tests.

    Returns pre-configured OutputT instances keyed by input hash.
    Eliminates API calls in unit tests.
    """
    def __init__(self, responses: dict[str, BaseModel] | BaseModel): ...
```

If a single `BaseModel` is passed, it always returns that response.
If a dict is passed, it looks up by input hash.

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
- `invoke()` with `FakeChatModel` returns expected output
- `ainvoke()` with `FakeChatModel` returns expected output
- `lazy=True` defers model creation (no error if config is invalid, until invoke)
- `lazy=True` builds chain on first invoke

Test cases for `FakeChatModel`:
- Single response mode always returns the same response
- Dict mode returns correct response for different inputs
- Missing key raises a clear error

---

## Notes

- `InputT` is bounded by `BaseModelKwargs` so `.to_kw()` can be used to extract
  field values for prompt rendering
- `OutputT` is a plain `BaseModel` - its schema is passed to `with_structured_output()`
- The chain is: `prompt_template | model.with_structured_output(output_model)`
- Both `invoke` and `ainvoke` call `.to_kw()` on the input to get the template variables dict

---

## Checklist

- [ ] Read laife's structured_chain.py for exact implementation
- [ ] Create StructuredLLMChain with bidirectional validation
- [ ] Add lazy initialization support
- [ ] Create FakeChatModel in testing module
- [ ] Create MissingPromptVariablesError, ExtraPromptVariablesError
- [ ] Write tests using FakeChatModel
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
