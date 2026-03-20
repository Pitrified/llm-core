# Block 2 - Chat config layer

Parent: [01-track-progress.md](01-track-progress.md)
Depends on: Block 1 (dependencies)

---

## Source reference

Extracted from laife: `src/laife/llm_services/chat/config/`

---

## Files to create

```
src/llm_core/chat/
    __init__.py
    config/
        __init__.py
        base.py             # ChatConfig + create_chat_model()
        openai.py           # ChatOpenAIConfig
        azure_openai.py     # AzureOpenAIChatConfig
        ollama.py           # OllamaChatConfig
        huggingface.py      # HuggingFaceChatConfig
```

---

## Design

### `ChatConfig(BaseModelKwargs)` - base.py

```python
class ChatConfig(BaseModelKwargs):
    model: str
    model_provider: str
    temperature: float = 0.2
    configurable_fields: ... | None = None  # if used by laife

    def create_chat_model(self) -> BaseChatModel:
        return init_chat_model(**self.to_kw(exclude_none=True))
```

Key points:
- `model_provider` is a plain string dispatched by LangChain - no enum needed
- `create_chat_model()` is the single factory; no direct constructor calls anywhere
- Provider subclasses override `model` and `model_provider` defaults, add provider-specific fields

### Provider subclasses

Each subclass:
- Sets `model` and `model_provider` defaults appropriate for that provider
- Adds provider-specific fields (e.g. `api_key: SecretStr`, `base_url`, `azure_endpoint`)
- Secret fields use `SecretStr`
- No logic beyond field defaults

**OpenAI:**
- `model_provider = "openai"`
- `model = "gpt-4o-mini"` (sensible default)
- `api_key: SecretStr | None = None` (falls back to env var)

**AzureOpenAI:**
- `model_provider = "azure_openai"`
- `azure_endpoint: str`
- `api_key: SecretStr | None = None`
- `openai_api_version: str`

**Ollama:**
- `model_provider = "ollama"`
- `model = "llama3.2"`
- `base_url: str = "http://localhost:11434"`

**HuggingFace:**
- `model_provider = "huggingface"`
- `model = "HuggingFaceH4/zephyr-7b-beta"` (example default)

---

## Tests

```
tests/chat/
    __init__.py
    config/
        __init__.py
        test_chat_config.py          # base config to_kw, field defaults
        test_openai.py               # OpenAI-specific fields
        test_azure_openai.py
        test_ollama.py
        test_huggingface.py
```

Test strategy:
- Construct each config, verify `to_kw()` output contains correct keys
- Verify `model_provider` is set correctly
- Verify `SecretStr` fields are masked in string output
- `create_chat_model()` tests are deferred to integration tests (need API keys)
  or use `FakeChatModel` once Block 5 ships it

---

## Checklist

- [ ] Read laife's actual chat config files for exact field names
- [ ] Create base.py with ChatConfig
- [ ] Create openai.py
- [ ] Create azure_openai.py
- [ ] Create ollama.py
- [ ] Create huggingface.py
- [ ] Create all __init__.py files
- [ ] Write tests
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
