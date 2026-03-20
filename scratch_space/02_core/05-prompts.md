# Block 4 - Prompt system

Parent: [01-track-progress.md](01-track-progress.md)
Depends on: Block 1 (dependencies)

---

## Source reference

Extracted from laife: `src/laife/llm/prompt_loader.py`

---

## Files to create

```
src/llm_core/prompts/
    __init__.py
    prompt_loader.py        # PromptLoader + PromptLoaderConfig
```

---

## Design

### `PromptLoaderConfig(BaseModelKwargs)`

```python
class PromptLoaderConfig(BaseModelKwargs):
    base_prompt_fol: Path          # root prompts folder in consumer project
    prompt_name: str               # subfolder name (e.g. "transcriber")
    version: str | int = "auto"    # "auto" = highest vN, or explicit int
```

### `PromptLoader`

```python
class PromptLoader:
    def __init__(self, config: PromptLoaderConfig) -> None: ...
    def load_prompt(self) -> str: ...
```

Behavior:
1. Resolve `base_prompt_fol / prompt_name /` directory
2. If `version == "auto"`: scan for `v*.jinja`, extract N, return highest
3. If `version` is an int: load `vN.jinja` directly
4. Read and return the template string
5. Cache in-memory (load once per PromptLoader instance)
6. Raise `NoPromptVersionFoundError` if no matching file exists

### File naming convention

```
<consumer_project>/prompts/<prompt_name>/v1.jinja
                                         v2.jinja   <- never edit; add v3 instead
```

The library ships the loader. Prompt files live in the consumer project.

---

## Exceptions

- `NoPromptVersionFoundError` - no `vN.jinja` found in the prompt directory

---

## Tests

```
tests/prompts/
    __init__.py
    test_prompt_loader.py
```

Test cases:
- Auto-version picks the highest `vN.jinja` from a temp directory
- Explicit version loads the correct file
- Missing prompt directory raises `NoPromptVersionFoundError`
- Missing specific version raises `NoPromptVersionFoundError`
- Non-sequential versions work (v1, v3, v5 - auto picks v5)
- Caching: second call to `load_prompt()` returns same object without re-reading
- Template content is returned as-is (no rendering at load time)

---

## Notes

- The prompt loader returns a raw Jinja2 template string, NOT a rendered prompt
- Rendering happens inside `StructuredLLMChain` via LangChain's `ChatPromptTemplate`
  with `template_format="jinja2"`
- Jinja2 is needed as a core dependency for `ChatPromptTemplate` to work with
  `template_format="jinja2"` - verify whether `langchain-core` already depends on
  it or if we need an explicit dep

---

## Checklist

- [ ] Read laife's prompt_loader.py for exact implementation
- [ ] Create PromptLoaderConfig
- [ ] Create PromptLoader with version discovery and caching
- [ ] Create NoPromptVersionFoundError
- [ ] Write tests with temp directories
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
