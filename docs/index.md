# llm-core

A reusable LLM tooling library for Python projects.

`llm-core` extracts and centralises the LLM integration patterns used across several projects
(laife, convo-craft, recipamatic) into a single, well-tested package. It provides:

- **Multi-provider chat configs** - OpenAI, Azure OpenAI, Ollama, HuggingFace via a unified
  `ChatConfig` interface that delegates to LangChain's `init_chat_model()`.
- **Embeddings configs** - same provider coverage, same `BaseModelKwargs` pattern.
- **`StructuredLLMChain[InputT, OutputT]`** - a generic dataclass that wires a Jinja2 prompt
  template to a chat model and enforces structured Pydantic output. Validates prompt variables
  at construction time. Supports both `invoke` and `ainvoke`.
- **Versioned Jinja2 prompts** - `PromptLoader` with `version="auto"` picks the highest `vN.jinja`
  file. Prompt files live in the consuming project; the loader is shipped here.
- **Vector store abstractions** - `VectorStoreConfig` → `ChromaConfig`; `CChroma` with SHA-256
  deduplication; `EntityStore` facade for `Vectorable` entities.
- **`Vectorable` protocol** - structural protocol for entity serialisation round-trips through a
  vector store (`to_document()` / `from_document()`).

## Quick Start

```bash
git clone https://github.com/pitrified/llm-core.git
cd llm-core
uv sync --group dev
uv run pytest
uv run mkdocs serve
```

## Project Structure

```
src/llm_core/
├── chat/          # ChatConfig + provider subclasses
├── embeddings/    # EmbeddingsConfig + provider subclasses
├── vectorstores/  # VectorStoreConfig, CChroma, EntityStore, Vectorable
├── chains/        # StructuredLLMChain[InputT, OutputT]
├── prompts/       # PromptLoader
├── data_models/   # BaseModelKwargs
├── metaclasses/   # Singleton
└── params/        # LlmCoreParams, LlmCorePaths, EnvType
```

## Next Steps

- [Getting Started](getting-started.md) - Set up your development environment
- [Guides](guides/uv.md) - Learn about the tools used in this project
- [Contributing](contributing.md) - How to contribute
