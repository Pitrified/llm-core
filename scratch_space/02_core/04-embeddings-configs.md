# Block 3 - Embeddings config layer

Parent: [01-track-progress.md](01-track-progress.md)
Depends on: Block 1 (dependencies)

---

## Source reference

Extracted from laife: `src/laife/llm_services/embeddings/config/`

---

## Files to create

```
src/llm_core/embeddings/
    __init__.py
    config/
        __init__.py
        base.py             # EmbeddingsConfig + create_embeddings()
        openai.py           # OpenAIEmbeddingsConfig
        azure_openai.py     # AzureOpenAIEmbeddingsConfig
        ollama.py           # OllamaEmbeddingsConfig
        huggingface.py      # HuggingFaceEmbeddingsConfig
```

---

## Design

### `EmbeddingsConfig(BaseModelKwargs)` - base.py

```python
class EmbeddingsConfig(BaseModelKwargs):
    model: str
    provider: str

    def create_embeddings(self) -> Embeddings:
        return init_embeddings(**self.to_kw(exclude_none=True))
```

Same dispatch pattern as chat: `init_embeddings()` routes on `provider` string.

### Provider subclasses

**OpenAI:**
- `provider = "openai"`
- `model = "text-embedding-3-small"` (cost-effective default)
- `api_key: SecretStr | None = None`

**AzureOpenAI:**
- `provider = "azure_openai"`
- `azure_endpoint: str`
- `api_key: SecretStr | None = None`

**Ollama:**
- `provider = "ollama"`
- `model = "nomic-embed-text"`
- `base_url: str = "http://localhost:11434"`

**HuggingFace:**
- `provider = "huggingface"`
- `model = "sentence-transformers/all-MiniLM-L6-v2"`

---

## Tests

```
tests/embeddings/
    __init__.py
    config/
        __init__.py
        test_embeddings_config.py
        test_openai.py
        test_azure_openai.py
        test_ollama.py
        test_huggingface.py
```

Same strategy as chat configs: test `to_kw()` output and field defaults.
Integration tests (actual embedding calls) gated by API key availability.

---

## Notes

- Embeddings config is consumed by vectorstore configs (Block 6) - ChromaConfig
  and PostgresVectorConfig both hold an `embeddings_config` field
- Keep the `provider` field name as-is (not `model_provider`) if that matches
  what `init_embeddings()` expects - verify against laife source

---

## Checklist

- [ ] Read laife's embeddings config files for exact field names
- [ ] Create base.py with EmbeddingsConfig
- [ ] Create provider subclasses
- [ ] Write tests
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
