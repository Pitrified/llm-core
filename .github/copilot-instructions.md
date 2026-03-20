# llm-core - Copilot Instructions

## Project overview

`llm-core` is a reusable LLM tooling library for Python projects. It centralises the LLM
integration patterns proven across laife, convo-craft, and recipamatic into a single package:
multi-provider chat/embeddings configs, a generic `StructuredLLMChain[InputT, OutputT]`,
versioned Jinja2 prompts, and vector store abstractions - all on top of LangChain and the
`BaseModelKwargs` config pattern. Python 3.14, managed with **uv**.

## Running & tooling

```bash
uv run pytest                        # run tests
uv run ruff check .                  # lint (ruff, ALL rules enabled)
uv run pyright                       # type-check (src/ and tests/ only)

uv run mkdocs serve                  # MkDocs local docs server
```

Credentials live at `~/cred/llm-core/.env` (loaded by `load_env()` in `src/llm_core/params/load_env.py`).

## Architecture layers

| Layer       | Path                                               | Role                                                                       |
| ----------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| Chat        | `src/llm_core/chat/config/`                        | `ChatConfig(BaseModelKwargs)` + OpenAI, Azure, Ollama, HuggingFace subs    |
| Embeddings  | `src/llm_core/embeddings/config/`                  | `EmbeddingsConfig` + same provider coverage                                |
| Vectorstore | `src/llm_core/vectorstores/`                       | `VectorStoreConfig`, `CChroma`, `EntityStore`, `Vectorable` protocol       |
| Chains      | `src/llm_core/chains/structured_chain.py`          | `StructuredLLMChain[InputT, OutputT]`                                      |
| Prompts     | `src/llm_core/prompts/prompt_loader.py`            | `PromptLoader` + `PromptLoaderConfig`; versioned `vN.jinja` files          |
| Data models | `src/llm_core/data_models/basemodel_kwargs.py`     | `BaseModelKwargs` - Pydantic base with `to_kw()` kwargs flattening         |
| Metaclasses | `src/llm_core/metaclasses/singleton.py`            | `Singleton` metaclass                                                      |
| Params      | `src/llm_core/params/llm_core_params.py`           | Singleton `LlmCoreParams`; aggregates paths and sample params              |
| Paths       | `src/llm_core/params/llm_core_paths.py`            | `LlmCorePaths`; env-aware filesystem references                            |
| Env type    | `src/llm_core/params/env_type.py`                  | `EnvStageType` (dev/prod) and `EnvLocationType` (local/render) enums       |

## Key patterns

**`LlmCoreParams` singleton**
Access project-wide config via `get_llm_core_params()` from `src/llm_core/params/llm_core_params.py`. It aggregates `LlmCorePaths` and `SampleParams`. Environment is controlled by `ENV_STAGE_TYPE` (`dev`/`prod`) and `ENV_LOCATION_TYPE` (`local`/`render`) env vars.

```python
from llm_core.params.llm_core_params import get_llm_core_params

params = get_llm_core_params()
paths = params.paths  # LlmCorePaths
```

**`ChatConfig` and provider subclasses**
All chat models share a `ChatConfig(BaseModelKwargs)` base with a `create_chat_model()` factory that calls LangChain's `init_chat_model(**self.to_kw(exclude_none=True))`. Provider subclasses only override `model`, `model_provider`, and provider-specific fields.

```python
from llm_core.chat.config.openai import ChatOpenAIConfig

chat = ChatOpenAIConfig(model="gpt-4o-mini").create_chat_model()
```

Same pattern for embeddings (`EmbeddingsConfig` + `create_embeddings()` via `init_embeddings`).

**`StructuredLLMChain[InputT, OutputT]`**
The core reusable primitive: a dataclass that wires a prompt template to a chat model and enforces structured Pydantic output. Input field names are validated against prompt variables at construction time.

```python
from llm_core.chains.structured_chain import StructuredLLMChain
from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
from pydantic import BaseModel

class MyInput(BaseModelKwargs):
    recipe_text: str   # must match {{ recipe_text }} in prompt

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
# async: await chain.ainvoke(...)
```

**Versioned Jinja2 prompts**
Prompts live in `<project>/prompts/<name>/vN.jinja`. Use `PromptLoader` with `version="auto"` to pick the highest version. Never edit an existing version file - add `vN+1.jinja` instead.

**`Vectorable` protocol**
Entities that round-trip through a vector store implement `to_document() -> Document` and
`from_document(doc) -> Self`. It is a structural `@runtime_checkable` protocol - no inheritance required.

**`BaseModelKwargs`**
Extend `BaseModelKwargs` (not plain `BaseModel`) for any config that needs to be forwarded as `**kwargs` to a third-party constructor. `to_kw(exclude_none=True)` flattens a nested `kwargs` dict at the top level.

**Config / Params separation**

- `src/llm_core/config/` holds Pydantic `BaseModelKwargs` models that define the _shape_ of settings. Use `SecretStr` for every sensitive field. Never read env vars inside config models.
- `src/llm_core/params/` holds plain classes that load _actual values_ and instantiate config models. Non-secret values are written as Python literals; env-switching is achieved via `match` on `env_type.stage` / `env_type.location`. Secrets are the only values loaded from `os.environ[VAR]` (raises `KeyError` naturally when missing).
- Every Params class accepts `env_type: EnvType | None = None` as its sole constructor argument. `__init__` only stores it and calls `_load_params()`. Loading is orchestrated via `_load_common_params()` then stage/location dispatch.
- Expose the assembled settings through `to_config()` returning the corresponding Pydantic model. Always mask secret fields in `__str__` using `[REDACTED]`.
- See `docs/guides/params_config.md` for the full reference with examples and common mistakes.

The canonical reference implementations are `src/llm_core/config/sample_config.py` and `src/llm_core/params/sample_params.py`.

**Env-aware paths**
`LlmCorePaths.load_config()` dispatches on `EnvLocationType` (`LOCAL` / `RENDER`) to set environment-specific paths. Common paths (`root_fol`, `cache_fol`, `data_fol`) are always set in `load_common_config_pre()`.

**`Singleton` metaclass**
Use `metaclass=Singleton` for any class that must have exactly one instance per process (e.g., `LlmCoreParams`). Reset in tests by clearing `Singleton._instances`.

## Style rules

- Never use em dashes (`--` or `---` or Unicode `—`). Use a hyphen `-` or rewrite the sentence.
- Use `loguru` (`from loguru import logger as lg`) for all logging.
- Raise descriptive custom exceptions (e.g., `UnknownEnvLocationError`) rather than bare `ValueError`/`RuntimeError`.

## Docstring style

Use **Google style** throughout. mkdocstrings is configured with `docstring_style: "google"`.

Standard sections use a label followed by a colon, with content indented by 4 spaces:

```python
def example(value: int) -> str:
    """One-line summary.

    Extended description as plain prose.

    Args:
        value: Description of the argument.

    Returns:
        Description of the return value.

    Raises:
        KeyError: If the key is missing.

    Example:
        Brief usage example::

            result = example(42)
    """
```

**Never use NumPy / Sphinx RST underline-style headers** (`Args\n----`, `Returns\n-------`, `Attributes\n----------`, etc.).

Rules:
- Section labels: `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Note:`, `Warning:`, `See Also:`, `Example:`, `Examples:` - always with a trailing colon, never with an underline.
- `Attributes:` in class docstrings uses two levels of indentation: the attribute name at +4 spaces, its description at +8 spaces.
- Module docstrings are narrative prose. Custom topic headings (e.g., "Pattern rules") are written as plain labelled paragraphs (`Pattern rules:`) - no underline, no RST heading markup.
- `See Also:` lists items as bare lines indented under the section label, not as `*` bullets.

## Testing & scratch space

- Tests live in `tests/` mirroring `src/llm_core/` structure.
- `scratch_space/` holds numbered exploratory notebooks and scripts. Not part of the package; ruff ignores `ERA001`/`F401`/`T20` there.

## Linting notes

- `ruff.toml` targets Python 3.13 with `select = ["ALL"]`. Key ignores: `COM812`, `D104`, `D203`, `D213`, `D413`, `FIX002`, `RET504`, `TD002`, `TD003`.
- Tests additionally allow `ARG001`, `INP001`, `PLR2004`, `S101`.
- Notebooks (`*.ipynb`) additionally allow `ERA001`, `F401`, `T20`.
- `meta/*` additionally allows `INP001`, `T20`.
- `max-args = 10` (pylint).

## End-of-task verification

After every code change, run the full verification suite before considering the task done:

```bash
uv run pytest && uv run ruff check . && uv run pyright
```
