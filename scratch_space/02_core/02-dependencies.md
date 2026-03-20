# Block 1 - Dependencies and framework alignment

Parent: [01-track-progress.md](01-track-progress.md)

**Status: DONE** - Implemented and verified (all tests pass, ruff clean, pyright 0 errors).

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
    "dotenv>=0.9.9",
    "jinja2>=3.1",
    "langchain>=0.3",
    "langchain-core>=0.3",
    "loguru>=0.7.3",
    "pydantic>=2.0",
]
```

- `pydantic` - BaseModelKwargs, all config/output models
- `langchain>=0.3` - full package required; `init_chat_model` / `init_embeddings` live in
  `langchain.chat_models` / `langchain.embeddings` (not in `langchain-core`)
- `langchain-core>=0.3` - abstractions (BaseChatModel, BaseEmbeddings, Document, ChatPromptTemplate)
- `jinja2` - PromptLoader template rendering
- `loguru` - logging
- `dotenv` - .env loading (already present)

### Optional dependency groups (provider-gated)

```toml
[project.optional-dependencies]
openai      = ["langchain-openai>=0.3"]
azure       = ["langchain-openai>=0.3"]
ollama      = ["langchain-ollama>=0.2"]
huggingface = ["langchain-huggingface>=0.1", "sentence-transformers>=3.0"]
chroma      = ["langchain-chroma>=0.1", "chromadb>=0.5"]
all         = ["llm-core[openai,azure,ollama,huggingface,chroma]"]
```

Note: `langchain-ollama>=0.2` (not 0.3) and `langchain-chroma>=0.1` (not 0.2) - adjusted
to match what the resolver successfully resolves to (latest: ollama 1.0.1, chroma 1.1.0).

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

## Open question - RESOLVED

Should `langchain` (full package) be a core dep, or just `langchain-core`?

**Answer: both are needed as core deps.**

- `init_chat_model` lives in `langchain.chat_models` (confirmed in laife source)
- `init_embeddings` lives in `langchain.embeddings` (confirmed in laife source)
- All other abstractions (BaseChatModel, Document, ChatPromptTemplate, etc.) come from
  `langchain-core`, which `langchain` depends on anyway
- Adding `langchain-core>=0.3` explicitly pins minimum abstraction version for consumers
  who need it directly

Resolved: `langchain>=0.3` + `langchain-core>=0.3` both listed as core deps.

---

## Resolved versions (uv sync --all-extras --all-groups)

| Package | Resolved |
|---|---|
| langchain | 1.2.13 |
| langchain-core | 1.2.20 |
| langchain-openai | 1.1.11 |
| langchain-ollama | 1.0.1 |
| langchain-huggingface | 1.2.1 |
| langchain-chroma | 1.1.0 |
| sentence-transformers | 5.3.0 |
