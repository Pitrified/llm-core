# Block 6 - Vectorstores

Parent: [01-track-progress.md](01-track-progress.md)
Depends on: Block 3 (embeddings configs)

---

## Source reference

Extracted from:
- laife: `src/laife/llm_services/vectorstores/` (CChroma, VectorStoreConfig, EntityStore)
- laife: `src/laife/entities/vectorable.py` (Vectorable protocol)
- recipinator: `backend/be/src/be/data/vector_db.py` (SHA-256 dedup, independently invented)

---

## Files to create

```
src/llm_core/vectorstores/
    __init__.py
    vectorable.py           # Vectorable protocol (@runtime_checkable)
    promptable.py           # Promptable protocol (per pitfall #10)
    hasher.py               # SHA-256 document ID generation
    config/
        __init__.py
        base.py             # VectorStoreConfig (abstract BaseModelKwargs)
        chroma.py           # ChromaConfig
    cchroma.py              # CChroma - dedup-aware Chroma wrapper
    entity_store.py         # EntityStore facade
```

---

## Design

### `Vectorable` protocol - vectorable.py

```python
@runtime_checkable
class Vectorable(Protocol):
    def to_document(self) -> Document: ...

    @classmethod
    def from_document(cls, doc: Document) -> Self: ...
```

- Structural protocol, not an ABC - no inheritance required
- Implementors must write `entity_type` into `doc.metadata`
- Per pitfall #7: add `validate_vectorable(obj)` helper that calls `to_document()`
  on a dummy instance and type-checks the result

### `Promptable` protocol - promptable.py (per pitfall #10)

```python
@runtime_checkable
class Promptable(Protocol):
    def to_prompt(self) -> str: ...
```

Lightweight protocol to enforce the `to_prompt()` convention. Optional
for consumers but enables type-safe chain input building.

### `hasher.py`

```python
def document_id(content: str, metadata: dict) -> str:
    """SHA-256 hash of content + sorted metadata as document ID."""
```

Deterministic, reproducible. Same content + metadata always produces the same ID.

### `VectorStoreConfig` - config/base.py

```python
class VectorStoreConfig(BaseModelKwargs, ABC):
    embeddings_config: EmbeddingsConfig

    @abstractmethod
    def create_store(self) -> VectorStore: ...
```

Abstract base for all vectorstore configs. Each backend implements `create_store()`.

### `ChromaConfig` - config/chroma.py

```python
class ChromaConfig(VectorStoreConfig):
    collection_name: str = "default"
    persist_directory: str | None = None

    def create_store(self) -> CChroma: ...
```

### `CChroma` - cchroma.py

```python
class CChroma(Chroma):
    """Dedup-aware Chroma wrapper.

    add_documents() hashes each doc via SHA-256 and skips duplicates.
    """
```

Per pitfall #9, consider whether to split this into:
- `DeduplicatingVectorStore` - wraps any LangChain vector store with SHA-256 dedup
- `CChroma(DeduplicatingVectorStore)` - Chroma-specific wrapper

Decision: start with `CChroma` directly (simpler); split later if a second
backend (Postgres) needs the same dedup logic.

### `EntityStore` - entity_store.py

```python
class EntityStore:
    """High-level facade for Vectorable entity persistence.

    Accepts Vectorable entities, calls to_document() / from_document().
    """
    def __init__(self, config: VectorStoreConfig) -> None: ...
    def save(self, entity: Vectorable) -> None: ...
    def save_many(self, entities: list[Vectorable]) -> None: ...
    def search(self, query: str, k: int = 4) -> list[Document]: ...
    def search_typed[T: Vectorable](
        self, query: str, entity_type: type[T], k: int = 4, **filter_kwargs
    ) -> list[T]: ...
```

---

## Tests

```
tests/vectorstores/
    __init__.py
    test_vectorable.py
    test_promptable.py
    test_hasher.py
    config/
        __init__.py
        test_chroma_config.py
    test_cchroma.py          # needs chroma optional dep
    test_entity_store.py     # needs chroma optional dep
```

Test strategy:
- `Vectorable` / `Promptable`: protocol conformance checks with dummy classes
- `hasher`: determinism, different inputs produce different IDs
- `ChromaConfig`: `to_kw()` output, field defaults
- `CChroma`: dedup on `add_documents()` (use in-memory Chroma)
- `EntityStore`: save/search round-trip with dummy Vectorable entity

Tests that need Chroma should be marked with `pytest.mark.skipif` if
`chromadb` is not installed, or gated by the `chroma` optional dep group.

---

## Checklist

- [ ] Read laife's vectorstore files for exact implementation
- [ ] Read laife's vectorable.py (already read above)
- [ ] Create Vectorable protocol
- [ ] Create Promptable protocol
- [ ] Create hasher.py
- [ ] Create VectorStoreConfig (abstract)
- [ ] Create ChromaConfig
- [ ] Create CChroma with dedup
- [ ] Create EntityStore facade
- [ ] Add validate_vectorable() helper
- [ ] Write tests
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
