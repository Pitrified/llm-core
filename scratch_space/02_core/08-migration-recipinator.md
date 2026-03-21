# Block 8 - Migration guide: recipinator

Parent: [08-release-migration.md](08-release-migration.md)

recipinator has no active LLM inference but contains a custom `VectorDB(Chroma)`
subclass in `backend/be/src/be/data/vector_db.py` with SHA-256-based document
deduplication - the same idea independently re-implemented in llm-core's `CChroma`.
The migration replaces `VectorDB` with `CChroma`, and optionally adds the
`Vectorable` protocol + `EntityStore` for a typed, facade-based API.

---

## Scope

### Files to update

| File | Change |
|---|---|
| `backend/be/pyproject.toml` | Add `llm-core[chroma]` dependency |
| `backend/be/src/be/data/vector_db.py` | Replace `VectorDB` with `CChroma` |
| Source files that import `VectorDB` | Update import paths |
| Entity classes (optional) | Implement `Vectorable` to use `EntityStore` |

---

## 1. Update `pyproject.toml`

```toml
# Poetry (recipinator uses Poetry)
[tool.poetry.dependencies]
llm-core = {git = "https://github.com/pitrified/llm-core", tag = "v0.1.0", extras = ["chroma"]}
```

Remove `chromadb`, `langchain-chroma`, and `langchain-community` if they are
only used for the vector DB (they come transitively from `llm-core[chroma]`).
Keep them if other code depends on them directly.

---

## 2. Replace `VectorDB` with `CChroma`

The existing `VectorDB(Chroma)` computes SHA-256 hashes in `add_documents()` to
skip duplicates. `CChroma` does exactly the same via `DeduplicatingMixin`.

```python
# Before - backend/be/src/be/data/vector_db.py
from langchain_chroma import Chroma

class VectorDB(Chroma):
    def add_documents(self, documents, id_in_metadata="", **kwargs):
        # ... manual SHA-256, get existing, filter, add ...

# Usage
db = VectorDB(
    collection_name="recipes",
    embedding_function=embeddings,
    persist_directory="/data/vectorstore",
)
db.add_documents(docs)
results = db.similarity_search("pasta", k=5)
```

```python
# After - replace the entire vector_db.py file
from llm_core.vectorstores import CChroma

# Usage - same constructor signature as Chroma / VectorDB
db = CChroma(
    collection_name="recipes",
    embedding_function=embeddings,
    persist_directory="/data/vectorstore",
)
db.add_documents(docs)           # deduplication is handled automatically
results = db.similarity_search("pasta", k=5)

# Or construct from config:
from llm_core.vectorstores.config import ChromaConfig
from llm_core.embeddings import HuggingFaceEmbeddingsConfig

config = ChromaConfig(
    collection_name="recipes",
    embeddings_config=HuggingFaceEmbeddingsConfig(),
    persist_directory="/data/vectorstore",
)
db = config.create_store()  # returns CChroma
```

---

## 3. Optional: adopt `Vectorable` + `EntityStore`

If recipinator has entity classes (recipes, ingredients, etc.) that are saved to
and retrieved from the vector store, adopting the `Vectorable` protocol gives a
typed, facade-based API.

### Step 1 - implement `Vectorable` on entity classes

```python
from typing import Self
from langchain_core.documents import Document

class Recipe:
    def __init__(self, name: str, text: str, recipe_id: str) -> None:
        self.name = name
        self.text = text
        self.recipe_id = recipe_id

    # Implement Vectorable (structural - no inheritance required)
    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "entity_type": "recipe",
                "name": self.name,
                "recipe_id": self.recipe_id,
            },
        )

    @classmethod
    def from_document(cls, doc: Document) -> Self:
        return cls(
            name=doc.metadata["name"],
            text=doc.page_content,
            recipe_id=doc.metadata["recipe_id"],
        )
```

### Step 2 - use `EntityStore`

```python
from llm_core.vectorstores import EntityStore
from llm_core.vectorstores.config import ChromaConfig
from llm_core.embeddings import HuggingFaceEmbeddingsConfig

store = EntityStore(ChromaConfig(
    collection_name="recipes",
    embeddings_config=HuggingFaceEmbeddingsConfig(),
    persist_directory="/data/vectorstore",
))

# Save
store.save(recipe)                  # single entity
store.save([recipe1, recipe2])      # multiple entities (deduplication automatic)

# Search - typed return
results: list[Recipe] = store.search("pasta with tomato", entity_type=Recipe, k=5)

# Search - raw Documents
docs = store.search("pasta with tomato", k=5)
```

---

## 4. `document_id` hash - compatibility note

recipinator's `get_document_id()` hashes content in 4 KB chunks and serializes
metadata via `json.dumps(sort_keys=True)`. llm-core's `document_id()` hashes the
full content in one call, then appends the same metadata JSON. **The hash values
will differ**, so existing stored document IDs will not match. If you have a
persisted Chroma collection, either:

- Clear and rebuild it after migration, or
- Keep `get_document_id()` alongside `CChroma` temporarily, passing pre-computed IDs
  via `db.add_documents(docs, ids=[get_document_id(d) for d in docs])`.

---

## 5. Verify

```bash
cd /home/pmn/repos/recipinator/backend/be
poetry run pytest   # or uv run pytest if migrated to uv
```
