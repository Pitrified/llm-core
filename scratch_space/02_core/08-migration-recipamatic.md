# Block 8 - Migration guide: recipamatic

Parent: [08-release-migration.md](08-release-migration.md)

recipamatic has a single LLM task: transcribing raw recipe text into a typed
`RecipeCore` hierarchy. The migration replaces its ad-hoc config and LCEL chain
with llm-core's `ChatOpenAIConfig` + `StructuredLLMChain`.

---

## Scope

The LLM integration lives in `py/src/recipamatic/cook/recipe_core/`. The
`RecipeCore` output model and all Pydantic sub-models are **not** changed.

### Files to update

| File | Change |
|---|---|
| `py/pyproject.toml` | Replace provider dep with `llm-core[openai]` |
| `py/src/recipamatic/cook/recipe_core/transcriber.py` (or similar chain file) | Replace config + LCEL chain with `StructuredLLMChain` |

---

## 1. Update `pyproject.toml`

recipamatic currently uses Poetry. Migrate to `uv` or add the dependency in
Poetry syntax:

```toml
# uv (pyproject.toml [project])
[project]
dependencies = [
    "llm-core[openai] @ git+https://github.com/pitrified/llm-core@v0.1.0",
    # ... other existing deps (fastapi, instaloader, etc.)
]

# Poetry (alternative, if staying on Poetry)
[tool.poetry.dependencies]
llm-core = {git = "https://github.com/pitrified/llm-core", tag = "v0.1.0", extras = ["openai"]}
```

---

## 2. Replace the inline chat config

recipamatic currently defines its own `ChatOpenAIConfig(BaseModel)` that is
passed via `.model_dump()` to `ChatOpenAI(...)` directly. Replace it entirely:

```python
# Before - ad-hoc config
from pydantic import BaseModel
class ChatOpenAIConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    api_key: SecretStr | None = ...

model = ChatOpenAI(**config.model_dump())

# After - llm-core config
from llm_core.chat import ChatOpenAIConfig

config = ChatOpenAIConfig(model="gpt-4o-mini", temperature=0.2)
# model is created internally by StructuredLLMChain
```

---

## 3. Replace the LCEL chain with `StructuredLLMChain`

The current pattern is:

```python
# Before
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

transcriber_prompt = ChatPromptTemplate.from_template("Extract recipe: {recipe}")
model = ChatOpenAI(**config.model_dump())
chain = transcriber_prompt | model.with_structured_output(RecipeCore)
result: RecipeCore = chain.invoke({"recipe": raw_text})
```

Replace with:

```python
# After
from llm_core.chains import StructuredLLMChain
from llm_core.chat import ChatOpenAIConfig
from llm_core.data_models import BaseModelKwargs
from pydantic import BaseModel

# Input model - field names must match {{ variable }} in the prompt
class TranscriberInput(BaseModelKwargs):
    recipe: str

# The prompt string uses Jinja2 syntax ({{ }} instead of { })
TRANSCRIBER_PROMPT = "Extract recipe details from the following text: {{ recipe }}"

chain = StructuredLLMChain(
    chat_config=ChatOpenAIConfig(model="gpt-4o-mini"),
    prompt_str=TRANSCRIBER_PROMPT,
    input_model=TranscriberInput,
    output_model=RecipeCore,  # existing Pydantic model - no changes needed
)

result: RecipeCore = chain.invoke(TranscriberInput(recipe=raw_text))
# async: result = await chain.ainvoke(TranscriberInput(recipe=raw_text))
```

**Important:** The prompt template format changes from LangChain `{variable}` to
Jinja2 `{{ variable }}`. Update all prompt strings. `StructuredLLMChain`
validates that `TranscriberInput` field names match prompt variables at
construction time - mismatches raise `MissingPromptVariablesError` or
`ExtraPromptVariablesError` immediately.

---

## 4. Optional: versioned prompt file

If the prompt template grows beyond a one-liner, move it out of the source file
into a versioned Jinja2 file:

```
py/src/recipamatic/prompts/recipe_transcriber/v1.jinja
```

```python
from llm_core.prompts import PromptLoader, PromptLoaderConfig
from pathlib import Path

loader = PromptLoader(PromptLoaderConfig(
    base_prompt_fol=Path(__file__).parent.parent / "prompts",
    prompt_name="recipe_transcriber",
    version="auto",
))

chain = StructuredLLMChain(
    chat_config=ChatOpenAIConfig(),
    prompt_str=loader.load_prompt(),
    input_model=TranscriberInput,
    output_model=RecipeCore,
)
```

---

## 5. Verify

```bash
cd /home/pmn/repos/recipamatic/py
uv run pytest
```
