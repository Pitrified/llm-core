# Changelog

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-03-21

Initial release. Extracts and centralises the LLM integration patterns
proven across laife, convo-craft, recipamatic, and recipinator into a single
reusable library.

### Added

**Foundation**
- `BaseModelKwargs` - Pydantic base with `to_kw(exclude_none=True)` kwargs
  flattening for forwarding config to third-party constructors.
- `Singleton` metaclass - one instance per process, reset-able in tests via
  `Singleton._instances`.
- `EnvType` + `EnvStageType` / `EnvLocationType` enums - `DEV`/`PROD` x
  `LOCAL`/`RENDER` environment dispatch. `EnvType.from_env_var()` reads
  `ENV_STAGE_TYPE` and `ENV_LOCATION_TYPE`.
- `LlmCoreParams` singleton + `LlmCorePaths` - project-wide config and
  env-aware filesystem paths. Access via `get_llm_core_params()`.
- `SampleParams` / `SampleConfig` - canonical reference implementations of
  the Config/Params pattern with full docstrings and guide.
- `load_env()` - loads `~/cred/llm-core/.env` via python-dotenv.

**Chat config layer** (`llm_core.chat`)
- `ChatConfig(BaseModelKwargs)` - base class with `create_chat_model()` via
  LangChain's `init_chat_model`.
- Provider subclasses: `ChatOpenAIConfig`, `AzureOpenAIChatConfig`,
  `OllamaChatConfig`, `HuggingFaceChatConfig`.
- Each subclass only overrides `model`, `model_provider`, and provider-specific
  fields (e.g. `api_key`, `azure_endpoint`).

**Embeddings config layer** (`llm_core.embeddings`)
- `EmbeddingsConfig(BaseModelKwargs)` - base class with `create_embeddings()`
  via LangChain's `init_embeddings`.
- Provider subclasses: `OpenAIEmbeddingsConfig`, `AzureOpenAIEmbeddingsConfig`,
  `OllamaEmbeddingsConfig`, `HuggingFaceEmbeddingsConfig`.

**Prompt system** (`llm_core.prompts`)
- `PromptLoader` + `PromptLoaderConfig` - versioned Jinja2 file discovery
  (`vN.jinja`). `version="auto"` selects the highest N. Results cached in-memory.
- `NoPromptVersionFoundError` - raised when no `vN.jinja` files exist.

**Structured chain** (`llm_core.chains`)
- `StructuredLLMChain[InputT, OutputT]` - dataclass wiring a Jinja2 prompt to
  a chat model with `with_structured_output`. Input field names are validated
  against prompt variables eagerly in `__post_init__`; the chat model is built
  lazily on first `invoke` / `ainvoke`.
- `MissingPromptVariablesError` / `ExtraPromptVariablesError` - early failure
  when input model fields and prompt variables do not match.

**Vectorstores** (`llm_core.vectorstores`)
- `Vectorable` - `@runtime_checkable` structural protocol; implement
  `to_document()` and `from_document()` to round-trip through a vector store.
- `Promptable` - `@runtime_checkable` structural protocol; implement
  `to_prompt()` for objects that feed into LLM prompts.
- `document_id(content, metadata)` - SHA-256 hash used for deduplication.
- `DeduplicatingMixin` - adds content+metadata deduplication to any LangChain
  `VectorStore` backend.
- `CChroma(DeduplicatingMixin, Chroma)` - dedup-aware Chroma wrapper.
- `VectorStoreConfig(BaseModelKwargs, ABC)` - abstract base for vector store
  configs with `create_store()`.
- `ChromaConfig` - supports in-memory (ephemeral), local-persistent, and
  remote-server Chroma deployments.
- `EntityStore` - high-level facade: `save(entity)` / `save_many(entities)`;
  `search(query)` returning `list[Document]` or `list[T]` via typed overloads.

**Testing utilities** (`llm_core.testing`)
- `FakeChatModel` - deterministic `BaseChatModel` subclass for unit tests;
  cycles through pre-loaded `AIMessage` responses.
- `FakeChatModelConfig` - `ChatConfig` subclass that creates a `FakeChatModel`.

**Public API exports**
- All submodule `__init__.py` files now expose `__all__` with the key public
  symbols, enabling clean `from llm_core.chains import StructuredLLMChain` style
  imports.

### Dependencies

Core (always installed):
- `langchain>=0.3`, `langchain-core>=0.3`
- `pydantic>=2.0`, `jinja2>=3.1`, `loguru>=0.7.3`, `dotenv>=0.9.9`

Optional extras:
- `openai` / `azure` - `langchain-openai>=0.3`
- `ollama` - `langchain-ollama>=0.2`
- `huggingface` - `langchain-huggingface>=0.1`, `sentence-transformers>=3.0`
- `chroma` - `langchain-chroma>=0.1`, `chromadb>=0.5`
- `all` - all of the above

### Test coverage

120 tests across all modules; 0 ruff lint errors; 0 pyright type errors.

[0.1.0]: https://github.com/pitrified/llm-core/releases/tag/v0.1.0
