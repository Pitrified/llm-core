# Track progress - llm-core project setup

Source plan: [00-llm-tools-plan.md](00-llm-tools-plan.md)

Track progress of each macro block in this file, and link to the companion sub plans for details;
Update this file as needed to reflect changes in the plan, and to track progress of the project setup.

---

## Current state

Blocks 0-6 are fully implemented and tested:

- `BaseModelKwargs` (data_models)
- `Singleton` metaclass
- `EnvType` system (DEV/PROD x LOCAL/RENDER)
- `LlmCoreParams` singleton, `LlmCorePaths`, `SampleParams`, `SampleConfig`
- `load_env()`
- `ChatConfig` + provider subclasses (OpenAI, AzureOpenAI, Ollama, HuggingFace)
- `EmbeddingsConfig` + provider subclasses (OpenAI, AzureOpenAI, Ollama, HuggingFace)
- `PromptLoader` + `PromptLoaderConfig` + `NoPromptVersionFoundError`
- `StructuredLLMChain[InputT, OutputT]` + `FakeChatModel` / `FakeChatModelConfig`
- `Vectorable` + `Promptable` protocols, `document_id` hasher
- `VectorStoreConfig`, `ChromaConfig`, `DeduplicatingMixin`, `CChroma`
- `EntityStore` facade (save / save_many / search with typed overload)
- Full test suite for all the above (119 tests, all pass)

**Not yet implemented:** Exceptions module, polish (Block 7), release (Block 8).

---

## Macro blocks

### Block 0 - Foundation scaffolding
**Status: DONE**

Everything under `params/`, `config/`, `data_models/`, `metaclasses/` is
implemented and tested. No further work needed.

---

### Block 1 - Dependencies and framework alignment
**Status: DONE**
**Sub-plan:** [02-dependencies.md](02-dependencies.md)

- Removed Haystack deps (`haystack-ai`, `chroma-haystack`, `ollama-haystack`, `openai`, `tiktoken`)
- Added core deps: `langchain>=0.3`, `langchain-core>=0.3`, `pydantic>=2.0`, `jinja2>=3.1`
- Added `[project.optional-dependencies]` groups: `openai`, `azure`, `ollama`, `huggingface`,
  `chroma`, `all`
- `uv sync --all-extras --all-groups` resolved cleanly (204 packages)
- `uv run pytest && uv run ruff check . && uv run pyright` - all 24 tests pass, 0 lint/type errors
- Open question resolved: `langchain>=0.3` (full package) is required as a core dep because
  `init_chat_model` / `init_embeddings` live in `langchain`, not `langchain-core`

---

### Block 2 - Chat config layer
**Status: DONE**
**Sub-plan:** [03-chat-configs.md](03-chat-configs.md)

- `src/llm_core/chat/config/base.py` - `ChatConfig(BaseModelKwargs)` + `create_chat_model()`
- Provider subclasses: OpenAI, AzureOpenAI, Ollama, HuggingFace
- Each provider only overrides `model`, `model_provider`, and provider-specific fields
- Uses `init_chat_model(**self.to_kw(exclude_none=True))` for dispatch
- Tests for each provider config (construction, `to_kw()` output; integration tests deferred)
- 25 tests, all pass; 0 ruff / pyright errors

---

### Block 3 - Embeddings config layer
**Status: DONE**
**Sub-plan:** [04-embeddings-configs.md](04-embeddings-configs.md)

- `src/llm_core/embeddings/config/base.py` - `EmbeddingsConfig(BaseModelKwargs)` + `create_embeddings()`
- Provider subclasses: OpenAI, AzureOpenAI, Ollama, HuggingFace
- Uses `init_embeddings(**self.to_kw(exclude_none=True))` for dispatch
- Note: field is `provider` (not `model_provider`) - matches what `init_embeddings()` expects
- 20 tests, all pass; 0 ruff / pyright errors

---

### Block 4 - Prompt system
**Status: DONE**
**Sub-plan:** [05-prompts.md](05-prompts.md)

- `src/llm_core/prompts/prompt_loader.py` - `PromptLoader` + `PromptLoaderConfig`
- Versioned Jinja2 file discovery (`vN.jinja`, `version="auto"` picks highest)
- In-memory caching (cache per `PromptLoader` instance)
- Custom exceptions: `NoPromptVersionFoundError`
- Note: `PromptLoaderConfig` extends `BaseModelKwargs` (not plain `BaseModel`) for consistency
- 12 tests, all pass; 0 ruff / pyright errors

---

### Block 5 - StructuredLLMChain
**Status: DONE**
**Sub-plan:** [06-structured-chain.md](06-structured-chain.md)

- `src/llm_core/chains/structured_chain.py` - `StructuredLLMChain[InputT, OutputT]` (dataclass)
- Property-based lazy init: `_chain` is `None` until `chain` property is first accessed
- Prompt variable validation (bidirectional) at construction via `__post_init__`
- `invoke()` and `ainvoke()` with Pydantic output type checking
- Custom exceptions: `MissingPromptVariablesError`, `ExtraPromptVariablesError`
- `FakeChatModel(BaseChatModel)` in `src/llm_core/testing/` with round-robin responses
- `FakeChatModelConfig(ChatConfig)` for use in tests
- 14 tests (8 chain + 6 fake model), all pass; 0 ruff / pyright errors

---

### Block 6 - Vectorstores
**Status: DONE**
**Sub-plan:** [07-vectorstores.md](07-vectorstores.md)

- `src/llm_core/vectorstores/vectorable.py` - `Vectorable` protocol (`@runtime_checkable`)
- `src/llm_core/vectorstores/promptable.py` - `Promptable` protocol (`@runtime_checkable`)
- `src/llm_core/vectorstores/hasher.py` - SHA-256 `document_id(content, metadata)` function
- `src/llm_core/vectorstores/config/base.py` - `VectorStoreConfig(BaseModelKwargs, ABC)`
- `src/llm_core/vectorstores/config/chroma.py` - `ChromaConfig` (ephemeral / persistent / server)
- `src/llm_core/vectorstores/cchroma.py` - `DeduplicatingMixin` + `CChroma(DeduplicatingMixin, Chroma)`
- `src/llm_core/vectorstores/entity_store.py` - `EntityStore` facade (save / save_many / overloaded search)
- 24 tests (3 vectorable + 3 promptable + 6 hasher + 3 chroma config + 4 cchroma + 5 entity store)
- All pass; 0 ruff / pyright errors

---

### Block 7 - Exceptions and polish
**Status: MERGED WITH RELEASE/SKIPPED/DEFERRED**

- Centralized `src/llm_core/exceptions.py` with all custom exceptions: skipped
- `validate_vectorable()` helper (per pitfall #7): deferred, add it to future steps
- Review all `__init__.py` public API exports: skipped
- Docstrings (Google style) on all public classes/methods: deferred to block 8
- Run full verification: `uv run pytest && uv run ruff check . && uv run pyright`: done

---

### Block 8 - v1 release and consumer migration
**Status: DONE**
**Sub-plan:** [08-release-migration.md](08-release-migration.md)

- [x] Write `CHANGELOG.md` (v0.1.0)
- [x] Git tag `v0.1.0`
- [x] Consumer migration guides: [laife](08-migration-laife.md),
      [recipamatic](08-migration-recipamatic.md),
      [convo-craft](08-migration-convo-craft.md),
      [recipinator](08-migration-recipinator.md)
- [x] README.md updated with install + full quickstart examples
- [x] All `__init__.py` files expose `__all__` with key public symbols
- [x] Docstrings complete (Google style); `to_kw()` Returns section added;
      `PromptLoader._prompt_path()` docstring added
- [x] 120 tests pass; 0 ruff; 0 pyright
- Makefile template for local editable development: deferred (see 08-release-migration.md)

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
