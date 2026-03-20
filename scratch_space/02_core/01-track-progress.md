# Track progress - llm-core project setup

Source plan: [00-llm-tools-plan.md](00-llm-tools-plan.md)

Track progress of each macro block in this file, and link to the companion sub plans for details;
Update this file as needed to reflect changes in the plan, and to track progress of the project setup.

---

## Current state

The foundation layers are implemented and tested:

- `BaseModelKwargs` (data_models)
- `Singleton` metaclass
- `EnvType` system (DEV/PROD x LOCAL/RENDER)
- `LlmCoreParams` singleton, `LlmCorePaths`, `SampleParams`, `SampleConfig`
- `load_env()`
- Full test suite for the above

**Not yet implemented:** chat configs, embeddings configs, StructuredLLMChain,
PromptLoader, vectorstores, exceptions module - i.e. all the LLM-specific layers.

**Dependency question:** `pyproject.toml` currently lists Haystack dependencies
(`haystack-ai`, `ollama-haystack`, `chroma-haystack`), but the plan and
`copilot-instructions.md` describe a LangChain-based architecture. This must
be resolved in Block 1.

---

## Macro blocks

### Block 0 - Foundation scaffolding
**Status: DONE**

Everything under `params/`, `config/`, `data_models/`, `metaclasses/` is
implemented and tested. No further work needed.

---

### Block 1 - Dependencies and framework alignment
**Status: NOT STARTED**
**Sub-plan:** [02-dependencies.md](02-dependencies.md)

- Decide Haystack vs LangChain (plan says LangChain; current deps say Haystack)
- Restructure `pyproject.toml` with optional dependency groups per provider
- Core: `pydantic`, `langchain-core`, `jinja2`, `loguru`
- Optional: `openai`, `azure`, `ollama`, `huggingface`, `chroma`
- Ensure `uv sync` resolves cleanly
- Run `uv run pytest && uv run ruff check . && uv run pyright` to verify

---

### Block 2 - Chat config layer
**Status: NOT STARTED**
**Sub-plan:** [03-chat-configs.md](03-chat-configs.md)

- `src/llm_core/chat/config/base.py` - `ChatConfig(BaseModelKwargs)` + `create_chat_model()`
- Provider subclasses: OpenAI, AzureOpenAI, Ollama, HuggingFace
- Each provider only overrides `model`, `model_provider`, and provider-specific fields
- Uses `init_chat_model(**self.to_kw(exclude_none=True))` for dispatch
- Tests for each provider config (construction, `to_kw()` output, `create_chat_model()`)

---

### Block 3 - Embeddings config layer
**Status: NOT STARTED**
**Sub-plan:** [04-embeddings-configs.md](04-embeddings-configs.md)

- `src/llm_core/embeddings/config/base.py` - `EmbeddingsConfig(BaseModelKwargs)` + `create_embeddings()`
- Provider subclasses: OpenAI, AzureOpenAI, Ollama, HuggingFace
- Uses `init_embeddings(**self.to_kw(exclude_none=True))` for dispatch
- Tests

---

### Block 4 - Prompt system
**Status: NOT STARTED**
**Sub-plan:** [05-prompts.md](05-prompts.md)

- `src/llm_core/prompts/prompt_loader.py` - `PromptLoader` + `PromptLoaderConfig`
- Versioned Jinja2 file discovery (`vN.jinja`, `version="auto"` picks highest)
- In-memory caching
- Custom exceptions: `NoPromptVersionFoundError`
- Tests with temp directories

---

### Block 5 - StructuredLLMChain
**Status: NOT STARTED**
**Sub-plan:** [06-structured-chain.md](06-structured-chain.md)

- `src/llm_core/chains/structured_chain.py` - `StructuredLLMChain[InputT, OutputT]`
- Prompt variable validation (both directions, per pitfall #6)
- `invoke()` and `ainvoke()`
- `lazy: bool = False` flag (per pitfall #5)
- `FakeChatModel` in `src/llm_core/testing/` (moved to v1 per pitfall #8)
- Custom exceptions: `MissingPromptVariablesError`, `ExtraPromptVariablesError`
- Tests using `FakeChatModel`

---

### Block 6 - Vectorstores
**Status: NOT STARTED**
**Sub-plan:** [07-vectorstores.md](07-vectorstores.md)

- `src/llm_core/vectorstores/vectorable.py` - `Vectorable` protocol (`@runtime_checkable`)
- `src/llm_core/vectorstores/hasher.py` - SHA-256 document ID generation
- `src/llm_core/vectorstores/config/base.py` - `VectorStoreConfig` (abstract)
- `src/llm_core/vectorstores/config/chroma.py` - `ChromaConfig`
- `src/llm_core/vectorstores/cchroma.py` - `CChroma` (dedup-aware Chroma wrapper)
- `src/llm_core/vectorstores/entity_store.py` - `EntityStore` facade
- Consider splitting `DeduplicatingVectorStore` from `EntityStore` (per pitfall #9)
- Optional `Promptable` protocol (per pitfall #10)
- Tests

---

### Block 7 - Exceptions and polish
**Status: NOT STARTED**

- Centralized `src/llm_core/exceptions.py` with all custom exceptions
- `validate_vectorable()` helper (per pitfall #7)
- Review all `__init__.py` public API exports
- Docstrings (Google style) on all public classes/methods
- Run full verification: `uv run pytest && uv run ruff check . && uv run pyright`

---

### Block 8 - v1 release and consumer migration
**Status: NOT STARTED**
**Sub-plan:** [08-release-migration.md](08-release-migration.md)

- Write `CHANGELOG.md` (v0.1.0)
- Git tag `v0.1.0`
- Consumer migration guide per repo (laife, recipamatic, convo-craft, recipinator)
- Makefile template for local editable development
- Update consumer `pyproject.toml` examples
- Document extension contract (subclassing for new providers)

---

## Future phases (post-v1)

These are tracked for awareness but not actionable until v1 is stable and
consumed by at least one project.

| Phase | Scope | Depends on |
|-------|-------|------------|
| 1.5 | PostgreSQL vectorstore (`pgvector` + SQLModel) | Block 6 |
| 2 | Conversation history and RAG chains | Blocks 5, 6 |
| 3 | Streaming (`stream()` / `astream()`) | Block 5 |
| 4 | Tool calling and agent loop | Block 5 |
| 5 | Observability, retry, rate limiting | Block 5 |
| 6 | Model evaluation and `ChainEval` harness | Blocks 5, 4 |

---

## Execution order and dependencies

```
Block 0 (DONE)
  |
Block 1 (deps)
  |
  +---> Block 2 (chat)
  |       |
  +---> Block 3 (embeddings) ---+
  |                              |
  +---> Block 4 (prompts)       |
          |                      |
          v                      v
        Block 5 (chain) <-------+
          |                      |
          v                      |
        Block 6 (vectorstores) --+
          |
          v
        Block 7 (polish)
          |
          v
        Block 8 (release)
```

Blocks 2, 3, and 4 are independent of each other and can be worked in parallel.
Block 5 depends on 2 and 4. Block 6 depends on 3. Block 7 depends on all prior.
