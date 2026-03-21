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

### `DeduplicatingMixin` and `CChroma` - cchroma.py

Deduplication logic lives in a single mixin class. Any LangChain `VectorStore` backend
that accepts `ids=` in `add_documents()` can gain dedup by listing the mixin first in
its MRO.

```python
# src/llm_core/vectorstores/cchroma.py

class DeduplicatingMixin:
    """Mixin that adds SHA-256-based deduplication to any LangChain VectorStore.

    Place before the concrete backend in the MRO so cooperative `super()` calls
    forward correctly to the backend's `add_documents()`.

    The backend must accept `ids` as a keyword argument to `add_documents()`.
    """

    def add_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        ids = [document_id(doc.page_content, doc.metadata) for doc in documents]
        existing = self._get_existing_ids(ids)        # backend-specific retrieval
        new_pairs = [(doc, id_) for doc, id_ in zip(documents, ids, strict=True)
                     if id_ not in existing]
        if not new_pairs:
            return []
        new_docs, new_ids = zip(*new_pairs, strict=True)
        return super().add_documents(list(new_docs), ids=list(new_ids), **kwargs)  # type: ignore[misc]

    def _get_existing_ids(self, ids: list[str]) -> frozenset[str]:
        """Return the subset of `ids` already present in the store."""
        # Default: check via get(); backends may override with a cheaper query.
        result = self.get(ids=ids)                    # type: ignore[attr-defined]
        return frozenset(result["ids"])


class CChroma(DeduplicatingMixin, Chroma):
    """Dedup-aware Chroma vector store.

    MRO: CChroma -> DeduplicatingMixin -> Chroma -> VectorStore
    `add_documents()` from the mixin intercepts, deduplicates, then calls
    `Chroma.add_documents()` via `super()`.
    """
```

**MRO walkthrough for `CChroma`:**
```
CChroma
  └─ DeduplicatingMixin   (add_documents: dedup, then super())
       └─ Chroma          (add_documents: actual Chromadb write)
            └─ VectorStore
```

**Extending to a second backend (Postgres example):**

When `pgvector` support is added (Phase 1.5), the same mixin is reused:

```python
# src/llm_core/vectorstores/cpgvector.py  (future)
from langchain_postgres import PGVector

class CPgVector(DeduplicatingMixin, PGVector):
    """Dedup-aware PGVector store.

    MRO: CPgVector -> DeduplicatingMixin -> PGVector -> VectorStore
    No dedup logic is duplicated; DeduplicatingMixin handles it all.
    """
```

The `DeduplicatingMixin._get_existing_ids()` default uses `.get(ids=ids)`, which works for
Chroma. If a backend does not support `.get()`, it overrides `_get_existing_ids()` in its
wrapper class - the mixin's `add_documents()` stays unchanged.

**Files to create / update:**

```
src/llm_core/vectorstores/
    deduplicating_mixin.py  # DeduplicatingMixin (NEW - extracted from cchroma.py)
    cchroma.py              # CChroma(DeduplicatingMixin, Chroma)
```

or keep `DeduplicatingMixin` co-located in `cchroma.py` for v1 (simpler); move to its own
file only when a second backend is added.

**Decision for v1:** co-locate in `cchroma.py`; add `cpgvector.py` in Phase 1.5 and move
`DeduplicatingMixin` to `deduplicating_mixin.py` at that point.

### `EntityStore` - entity_store.py

```python
class EntityStore:
    """High-level facade for Vectorable entity persistence.

    Accepts Vectorable entities, calls to_document() / from_document().
    """
    def __init__(self, config: VectorStoreConfig) -> None: ...
    def save(self, entity: Vectorable) -> None: ...
    def save_many(self, entities: list[Vectorable]) -> None: ...

    @overload
    def search(self, query: str, *, k: int = ..., **filter_kwargs: Any) -> list[Document]: ...
    @overload
    def search[T: Vectorable](
        self, query: str, *, entity_type: type[T], k: int = ..., **filter_kwargs: Any
    ) -> list[T]: ...
    def search(
        self,
        query: str,
        *,
        entity_type: type[Vectorable] | None = None,
        k: int = 4,
        **filter_kwargs: Any,
    ) -> list[Document] | list[Vectorable]:
        docs = self._store.similarity_search(query, k=k, filter=filter_kwargs or None)
        if entity_type is None:
            return docs
        return [entity_type.from_document(doc) for doc in docs]
```

`search_typed()` is removed. The single `search()` method covers both cases via overloads:
- `store.search("query")` returns `list[Document]`
- `store.search("query", entity_type=MyEntity)` returns `list[MyEntity]`

`entity_type` is keyword-only (after `*`) to prevent accidental positional ordering bugs.

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
- `EntityStore.search()` without `entity_type` returns `list[Document]`
- `EntityStore.search()` with `entity_type` returns typed list via `from_document()`
- `EntityStore.search()` with `entity_type` and `filter_kwargs` passes filter to backend

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
- [ ] Create `DeduplicatingMixin` (co-located in `cchroma.py` for v1)
- [ ] Create `CChroma(DeduplicatingMixin, Chroma)` with dedup
- [ ] Create EntityStore facade with overloaded `search()` (drop `search_typed()`)
- [ ] Add validate_vectorable() helper
- [ ] Write tests
- [ ] `uv run pytest && uv run ruff check . && uv run pyright`
