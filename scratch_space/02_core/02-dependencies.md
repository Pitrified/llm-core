# Block 1 - Dependencies and framework alignment

Parent: [01-track-progress.md](01-track-progress.md)

---

## Problem

`pyproject.toml` currently declares Haystack-based dependencies:

```toml
dependencies = [
    "chroma-haystack>=2.0.1",
    "haystack-ai>=2.10.0",
    "ollama-haystack>=2.3.0",
    "openai>=1.63.0",
    "tiktoken>=0.9.0",
    "dotenv>=0.9.9",
    "loguru>=0.7.3",
]
```

The plan, `copilot-instructions.md`, and the laife source code all describe a
LangChain-based architecture (`langchain-core`, `init_chat_model`, `init_embeddings`,
`ChatPromptTemplate`, `with_structured_output`). The Haystack deps are leftovers
from the template and must be replaced.

---

## Target dependency structure

### Core (always installed)

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "langchain-core>=0.3",
    "jinja2>=3.1",
    "loguru>=0.7.3",
    "dotenv>=0.9.9",
]
```

- `pydantic` - BaseModelKwargs, all config/output models
- `langchain-core` - abstractions only (BaseChatModel, BaseEmbeddings, Document, ChatPromptTemplate)
- `jinja2` - PromptLoader template rendering
- `loguru` - logging
- `dotenv` - .env loading (already present)

### Optional dependency groups (provider-gated)

```toml
[project.optional-dependencies]
openai      = ["langchain-openai>=0.3"]
azure       = ["langchain-openai>=0.3"]
ollama      = ["langchain-ollama>=0.3"]
huggingface = ["langchain-huggingface>=0.1", "sentence-transformers>=3.0"]
chroma      = ["langchain-chroma>=0.2", "chromadb>=0.5"]
all         = ["llm-core[openai,azure,ollama,huggingface,chroma]"]
```

### Dev groups (keep existing, update as needed)

Keep `test`, `lint`, `notebook`, `docs` groups as-is. No changes needed.

---

## Steps

1. Remove Haystack deps: `haystack-ai`, `chroma-haystack`, `ollama-haystack`, `openai`, `tiktoken`
2. Add core deps: `pydantic`, `langchain-core`, `jinja2`
3. Add optional dependency groups per provider
4. Run `uv sync --all-extras --all-groups` to verify resolver
5. Run `uv run pytest && uv run ruff check . && uv run pyright` - existing tests should still pass
   (they don't touch any LLM code)

---

## Version pin strategy

- Use wide lower-bound pins (`>=X.Y`) with no upper bound
- Never pin exact versions in a library - let consumers tighten if needed
- LangChain releases frequently; tight pins cause resolver conflicts downstream

---

## Open question

Should `langchain` (full package) be a core dep, or just `langchain-core`?

- `init_chat_model` lives in `langchain.chat_models` (the full `langchain` package)
- `init_embeddings` lives in `langchain.embeddings`
- If we only use `langchain-core`, we need to find the correct import paths
  or accept `langchain>=0.3` as a core dep

Decision: check laife imports to determine the minimum required package.
