# Block 8 - Migration guide: laife

Parent: [08-release-migration.md](08-release-migration.md)

laife contains the canonical implementation of every pattern that was extracted
into llm-core. The migration is therefore a near-mechanical substitution of
local copies with imports from the library.

---

## Scope

Replace the entire `src/laife/llm_services/` subtree and the three
domain-agnostic files in `src/laife/llm/` and `src/laife/entities/`.
Domain-specific files (brain, mission, planners, prompts) are **not** touched.

### Files to delete

```
src/laife/llm_services/chat/config/base.py
src/laife/llm_services/chat/config/chat_openai.py
src/laife/llm_services/chat/config/azure_openai.py
src/laife/llm_services/chat/config/ollama.py
src/laife/llm_services/chat/config/huggingface.py
src/laife/llm_services/chat/config/__init__.py
src/laife/llm_services/chat/__init__.py
src/laife/llm_services/embeddings/config/base.py
src/laife/llm_services/embeddings/config/openai.py
src/laife/llm_services/embeddings/config/azure_openai.py
src/laife/llm_services/embeddings/config/ollama.py
src/laife/llm_services/embeddings/config/huggingface.py
src/laife/llm_services/embeddings/config/__init__.py
src/laife/llm_services/embeddings/__init__.py
src/laife/llm_services/vectorstores/  (entire subtree)
src/laife/llm_services/__init__.py
src/laife/llm/structured_chain.py
src/laife/llm/prompt_loader.py
src/laife/entities/vectorable.py
```

### Files to keep (domain-specific, no changes needed)

```
src/laife/llm/player_brain.py
src/laife/llm/mission.py
src/laife/llm/mission_generator.py
src/laife/llm/player_planner.py
src/laife/llm/player_replier.py
src/laife/prompts/**/*.jinja
src/laife/entities/  (all except vectorable.py)
```

---

## 1. Update `pyproject.toml`

Remove the individual LangChain provider packages (they come transitively from
llm-core's extras), add the llm-core dependency:

```toml
[project]
dependencies = [
    "llm-core[all] @ git+https://github.com/pitrified/llm-core@v0.1.0",
    "loguru>=0.7.2",
    "pygame>=2.6.0",
    "pysqlite3-binary>=0.5.3",
    "rich>=13.9.2",
    # keep any deps not covered by llm-core[all]
]
```

Run `uv sync` to pull in the new dependency.

---

## 2. Update import paths

All code that imports from `laife.llm_services` or from the deleted files needs
its import paths updated. The mapping is one-to-one:

| Old import | New import |
|---|---|
| `from laife.llm_services.chat.config.base import ChatConfig` | `from llm_core.chat.config.base import ChatConfig` |
| `from laife.llm_services.chat.config.chat_openai import ChatOpenAIConfig` | `from llm_core.chat.config.openai import ChatOpenAIConfig` |
| `from laife.llm_services.chat.config.azure_openai import AzureOpenAIChatConfig` | `from llm_core.chat.config.azure_openai import AzureOpenAIChatConfig` |
| `from laife.llm_services.chat.config.ollama import OllamaChatConfig` | `from llm_core.chat.config.ollama import OllamaChatConfig` |
| `from laife.llm_services.chat.config.huggingface import HuggingFaceChatConfig` | `from llm_core.chat.config.huggingface import HuggingFaceChatConfig` |
| `from laife.llm_services.embeddings.config.base import EmbeddingsConfig` | `from llm_core.embeddings.config.base import EmbeddingsConfig` |
| `from laife.llm_services.embeddings.config.openai import OpenAIEmbeddingsConfig` | `from llm_core.embeddings.config.openai import OpenAIEmbeddingsConfig` |
| `from laife.llm.structured_chain import StructuredLLMChain` | `from llm_core.chains.structured_chain import StructuredLLMChain` |
| `from laife.llm.structured_chain import MissingPromptVariablesError` | `from llm_core.chains.exceptions import MissingPromptVariablesError` |
| `from laife.llm.prompt_loader import PromptLoader, PromptLoaderConfig` | `from llm_core.prompts.prompt_loader import PromptLoader, PromptLoaderConfig` |
| `from laife.llm.prompt_loader import NoPromptVersionFoundError` | `from llm_core.prompts.prompt_loader import NoPromptVersionFoundError` |
| `from laife.entities.vectorable import Vectorable` | `from llm_core.vectorstores.vectorable import Vectorable` |

Or use the submodule shorthand from `__init__`:

```python
from llm_core.chat import ChatOpenAIConfig, OllamaChatConfig
from llm_core.chains import StructuredLLMChain
from llm_core.prompts import PromptLoader, PromptLoaderConfig
from llm_core.vectorstores import Vectorable, EntityStore, CChroma
from llm_core.vectorstores.config import ChromaConfig
```

---

## 3. Behavior differences to watch for

### `StructuredLLMChain` - lazy vs eager chain construction

laife's version eagerly builds the full LCEL chain in `__post_init__`,
including calling `create_chat_model()`. This means object construction
requires a valid API key or a running Ollama server.

llm-core's version defers `create_chat_model()` to the first `invoke` /
`ainvoke` call. Construction only validates prompt variables (no API call).

**Impact:** tests that patched `self.chain` directly on the laife version
need to be updated. Use `llm_core.testing.FakeChatModelConfig` instead:

```python
from langchain_core.messages import AIMessage
from llm_core.testing import FakeChatModelConfig

fake_config = FakeChatModelConfig(
    responses=[AIMessage(content='{"action": "move", "direction": "north"}')]
)
chain = StructuredLLMChain(
    chat_config=fake_config,
    prompt_str="...",
    input_model=MyInput,
    output_model=MyOutput,
)
```

### `StructuredLLMChain` - bidirectional variable validation

laife's version only checks for missing variables (input model fields absent
from the prompt). llm-core's version also raises `ExtraPromptVariablesError`
when the prompt references variables not in the input model. Fix mismatches
before migration.

### `PromptLoaderConfig` - base class change

laife's `PromptLoaderConfig` extends plain `BaseModel`. llm-core's extends
`BaseModelKwargs`. All existing usage remains compatible; this only matters
if you call `.to_kw()` on the config itself (rare).

---

## 4. Vectorstores

If laife uses `EntityStore` / `CChroma` from its own `llm_services/vectorstores/`,
the migration is a drop-in import replacement. Entity classes that already
implement `to_document()` / `from_document()` satisfy the `Vectorable` protocol
without any changes.

Replace the `ChromaConfig` creation:

```python
# Before
from laife.llm_services.vectorstores.config.chroma import ChromaConfig
config = ChromaConfig(persist_directory=str(paths.vectorstore_fol))

# After
from llm_core.vectorstores.config import ChromaConfig
config = ChromaConfig(persist_directory=str(paths.vectorstore_fol))
```

---

## 5. Verify

```bash
cd /home/pmn/repos/laife
uv run pytest && uv run ruff check . && uv run pyright
```
