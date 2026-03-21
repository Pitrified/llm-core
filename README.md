# llm-core

Reusable LLM tooling library for Python projects. Provides multi-provider chat/embeddings
configs, a generic structured-output chain, versioned Jinja2 prompts, and vector store
abstractions - all built on LangChain and the `BaseModelKwargs` config pattern.

## Installation

Install from the git tag using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
or `pip`:

```bash
# OpenAI only
uv pip install "llm-core[openai] @ git+https://github.com/pitrified/llm-core@v0.1.0"

# All providers + Chroma
uv pip install "llm-core[all] @ git+https://github.com/pitrified/llm-core@v0.1.0"
```

For development (all extras + dev tools):

```bash
uv sync --all-extras --all-groups
```

## Quickstart

### Structured output chain

```python
from llm_core.chains import StructuredLLMChain
from llm_core.chat import ChatOpenAIConfig
from llm_core.data_models import BaseModelKwargs
from pydantic import BaseModel

class RecipeInput(BaseModelKwargs):
    recipe_text: str  # field name must match {{ recipe_text }} in prompt

class RecipeOutput(BaseModel):
    name: str
    ingredients: list[str]

chain = StructuredLLMChain(
    chat_config=ChatOpenAIConfig(model="gpt-4o-mini"),
    prompt_str="Extract recipe details from: {{ recipe_text }}",
    input_model=RecipeInput,
    output_model=RecipeOutput,
)
result: RecipeOutput = chain.invoke(RecipeInput(recipe_text="Boil pasta..."))
# async: result = await chain.ainvoke(...)
```

### Versioned prompts

```python
from llm_core.prompts import PromptLoader, PromptLoaderConfig
from pathlib import Path

loader = PromptLoader(PromptLoaderConfig(
    base_prompt_fol=Path("src/myproject/prompts"),
    prompt_name="recipe_extractor",  # loads src/myproject/prompts/recipe_extractor/v1.jinja
    version="auto",                  # picks the highest vN.jinja
))
prompt_str = loader.load_prompt()
```

### Switching providers

All chat and embeddings configs share the same interface - swap providers by
changing one import:

```python
from llm_core.chat import OllamaChatConfig, AzureOpenAIChatConfig

# Local Ollama
chain = StructuredLLMChain(chat_config=OllamaChatConfig(model="llama3.2"), ...)

# Azure OpenAI
chain = StructuredLLMChain(chat_config=AzureOpenAIChatConfig(model="gpt-4o"), ...)
```

### Vector store entity persistence

```python
from llm_core.vectorstores import EntityStore, Vectorable
from llm_core.vectorstores.config import ChromaConfig
from llm_core.embeddings import OpenAIEmbeddingsConfig
from langchain_core.documents import Document
from typing import Self

class MyEntity:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_document(self) -> Document:
        return Document(page_content=self.text, metadata={"entity_type": "my_entity"})

    @classmethod
    def from_document(cls, doc: Document) -> Self:
        return cls(text=doc.page_content)

store = EntityStore(ChromaConfig(
    embeddings_config=OpenAIEmbeddingsConfig(),
    persist_directory="/data/vectorstore",
))
store.save(MyEntity("Hello world"))
results: list[MyEntity] = store.search("hello", entity_type=MyEntity)
```

## Docs

Docs are available at [https://pitrified.github.io/llm-core/](https://pitrified.github.io/llm-core/).

## Setup

### Environment Variables

Create `~/cred/llm-core/.env` with the required keys (see `nokeys.env` for the full list).

For VSCode to pick up the env file, add to your workspace settings:

```json
"python.envFile": "/home/${env:USER}/cred/llm-core/.env"
```

### Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Linting and type checking

```bash
uv run ruff check --fix
uv run ruff format
uv run pyright
```

### Testing

```bash
uv run pytest
```
