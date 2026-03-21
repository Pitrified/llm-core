# Condition System to filter documents

## overview

### Feature description

Add support for more complex conditions (e.g., nested conditions, logical operators).

which we have somewhere (`laife/src/laife/llm_services/vectorstores/cond.py`)
search for similar patterns in the other repos as well

analyze existing,
check how to make them more flexible and composable, and then implement the new design in `llm_core` and migrate the existing code to use it.

#### NOTE1:

plan for multiple backends, chroma is used now but we want to be able to swap it out in the future
so keeping a superset of chroma's filter grammar is ok, as other backend might have more features, but we should avoid chroma-specific concepts leaking into the design (e.g., the `$eq` shorthand).
we need to design an agnostic condition system
eg: https://qdrant.tech/documentation/search/filtering/ is incredibly powerful, and we don't want to limit ourselves to chroma's simpler grammar.

#### NOTE2:

`EntityStore` must not touch serialization at all. It receives a `Cond` and passes it
straight to the vector store adapter via a protocol. The adapter (`CChroma`, future
`CQdrant`) is tied to one backend, so it knows how to serialize the `Cond` and how to
call `similarity_search` with the correct kwargs - including Chroma-specific ones like
`where_document` that do not exist on the abstract `VectorStore` base.

### migration plans

update migration plan
`llm-core/scratch_space/02_core/08-migration-laife.md`

## Brainstorm

## Codebase Overview

**Existing condition system** - `laife/src/laife/llm_services/vectorstores/cond.py`:

Three plain Python classes, each implementing a `to_dict()` method that produces
Chroma's NoSQL filter format:

- `CompCond(field, CompOp, value)` - comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- `InclusionCond(field, InclusionOp, *values)` - inclusion: `$in`, `$nin`
- `LogicalCond(LogicalOp, *conditions)` - logical: `$and`, `$or` (requires 2+ children)

Status: tested in `laife/tests/llm/test_cond.py`, not integrated into `EntityStore`.

**llm-core's `EntityStore`** - `src/llm_core/vectorstores/entity_store.py`:

Uses `**filter_kwargs` raw pass-through to `similarity_search(filter=...)`. No typed
condition support.

**Filter grammars across backends:**

| Concept        | Chroma                               | Qdrant                                        |
| -------------- | ------------------------------------ | --------------------------------------------- |
| Equality       | `{"field": value}` or `{"$eq": v}`   | `FieldCondition(..., match=MatchValue(v))`    |
| Comparison     | `{"field": {"$gt": v}}`              | `FieldCondition(..., range=Range(gt=v))`      |
| Inclusion      | `{"field": {"$in": [v1, v2]}}`       | `MatchAny(any=[v1, v2])`                      |
| NOT            | `$ne` / `$nin` per-field only        | `Filter(must_not=[...])`                      |
| AND / OR       | `{"$and": [...]}` / `{"$or": [...]}` | `Filter(must=[...])` / `Filter(should=[...])` |
| Document-level | `where_document={"$contains": "x"}`  | `FullTextMatch(text="x")`                     |

Chroma's grammar is a subset of Qdrant's. The AST must be rich enough to express
Qdrant-level semantics so nothing is lost when switching backends.

No other repo (`recipamatic`, `convo_craft`, `recipinator`) uses vector stores with
conditions.

---

## Eliminated options

**Option 1 (port laife as-is)**, **Option 3 (factory functions returning dicts)**, and
**Option A (standalone serializer classes injected into EntityStore)** are dropped.
Options 1 and 3 couple the public API to Chroma's wire format. Option A requires
injecting a serializer into `EntityStore`, which is noisy and contradicts NOTE2 -
`EntityStore` must not know anything about serialization.

---

## Option B - Backend-agnostic AST + `CondSearchable` protocol on the adapter

Pure-data AST (no Chroma concepts). Each vector store adapter implements a
`CondSearchable` protocol exposing a single `search(query, cond, k)` method. `EntityStore`
calls that method and is completely unaware of serialization. The adapter handles both
the condition serialization and the correct `similarity_search` kwargs internally.

```python
# llm_core/vectorstores/cond.py  (pure data, no backend concepts)
LiteralValue = str | int | float | bool

class CompOp(Enum):
    EQ = "eq"; NE = "ne"; GT = "gt"; GTE = "gte"; LT = "lt"; LTE = "lte"

class InclusionOp(Enum):
    IN = "in"; NIN = "nin"

class LogicalOp(Enum):
    AND = "and"; OR = "or"

@dataclass
class CompCond:
    field: str; op: CompOp; value: LiteralValue

@dataclass
class InclusionCond:
    field: str; op: InclusionOp; values: list[LiteralValue]

@dataclass
class NotCond:
    condition: "AnyCond"

@dataclass
class LogicalCond:
    op: LogicalOp; children: list["AnyCond"]

@dataclass
class DocCond:
    """Document-level text filter (e.g. contains / not_contains)."""
    text: str; negate: bool = False

AnyCond = CompCond | InclusionCond | NotCond | LogicalCond | DocCond

# llm_core/vectorstores/cond.py  (protocol)
class CondSearchable(Protocol):
    def search(self, query: str, k: int, cond: AnyCond | None) -> list[Document]: ...

# entity_store.py  (totally agnostic of serialization)
class EntityStore:
    _store: CondSearchable
    def search(self, query: str, *, cond: AnyCond | None = None, k: int = 4) -> list[...]:
        return self._store.search(query, k=k, cond=cond)

# cchroma.py  (owns both dedup and condition handling)
class CChroma(DeduplicatingMixin, Chroma):
    def search(self, query: str, k: int, cond: AnyCond | None) -> list[Document]:
        meta_filter, doc_filter = self._serialize(cond)  # private
        return self.similarity_search(
            query, k=k,
            filter=meta_filter,          # Chroma-specific kwarg
            where_document=doc_filter,   # Chroma-specific kwarg; None → omitted
        )

# future: cqdrant.py
class CQdrant(...):
    def search(self, query: str, k: int, cond: AnyCond | None) -> list[Document]:
        qdrant_filter = self._serialize(cond)  # produces qdrant Filter object
        return self.similarity_search(query, k=k, filter=qdrant_filter)
```

**Pros**

- `EntityStore` is fully agnostic - zero serialization logic, no backend-specific kwargs.
- No extra constructor argument anywhere.
- `CChroma` is a complete backend adapter; owning condition serialization alongside dedup
  is consistent with that role.
- Each adapter is independently testable via its `search()` method with a real or
  in-memory Chroma/Qdrant instance.
- The `CondSearchable` protocol decouples `EntityStore` from any concrete adapter class.
- `DocCond` handling stays inside the adapter: Chroma passes it as `where_document`,
  Qdrant uses `FullTextMatch` - this difference never surfaces in `EntityStore`.

**Cons**

- Serialization logic is not unit-testable without instantiating the adapter. Acceptable
  since `CChroma` already requires Chroma in its existing tests.

---

## Cross-cutting: operator overloading with smart flattening

Applies to both options. Each AST node class gains `__and__`, `__or__`, and `__invert__`
methods. The key implementation detail: when combining two nodes of the same logical
type, merge their children instead of nesting.

```python
class _CondMixin:
    def __and__(self, other: AnyCond) -> LogicalCond:
        # Flatten: (A & B) & C → AND(A, B, C)
        left = self.children if isinstance(self, LogicalCond) and self.op == LogicalOp.AND else [self]
        right = other.children if isinstance(other, LogicalCond) and other.op == LogicalOp.AND else [other]
        return LogicalCond(LogicalOp.AND, left + right)

    def __or__(self, other: AnyCond) -> LogicalCond:
        left = self.children if isinstance(self, LogicalCond) and self.op == LogicalOp.OR else [self]
        right = other.children if isinstance(other, LogicalCond) and other.op == LogicalOp.OR else [other]
        return LogicalCond(LogicalOp.OR, left + right)

    def __invert__(self) -> NotCond:
        return NotCond(self)

# a & b & c → AND(a, b, c)  not AND(AND(a,b), c)
# ~cond     → NotCond(cond) → serializer rewrites to backend-native form
```

Double-negation is eliminated at the AST level inside `__invert__` itself - no
serializer involvement needed:

```python
def __invert__(self) -> AnyCond:
    # ~~cond → cond
    return self.condition if isinstance(self, NotCond) else NotCond(self)
```

`~~cond` returns the original node unchanged. Serializers never see a `NotCond(NotCond(x))`.

---

## Resolved design decisions

1. **`$eq` shorthand** - removed. The AST uses enum values (`CompOp.EQ`), not wire
   strings. Each serializer decides the wire representation independently. No shorthand
   leaks into the shared model.

2. **`DocCond`** - included. Maps to `where_document` in Chroma and `FullTextMatch` in
   Qdrant. Single class with a `negate: bool` flag covers `$contains` and
   `$not_contains`.

3. **`NOT` operator** - included as a first-class `NotCond` node. Chroma serializers
   push negation down to the leaf (`NE`, `NIN`); Qdrant uses `must_not`. The difference
   is encapsulated in the serializer.

4. **`**filter_kwargs`** - removed outright from `EntityStore.search()`. It was always
untyped and undiscoverable. No deprecation period; the new `cond` parameter is the
   replacement.

5. **Minimum children on `LogicalCond`** - relaxed to 1 (or even 0 treated as a no-op).
   A single-child `AND` is vacuously valid and simplifies smart flattening.

---

## Recommendation

**Option B** (AST + `CondSearchable` protocol) with smart-flattening operator overloading
and AST-level double-negation elimination is the recommended design.

Rationale:

- `EntityStore` is a pure facade - it never sees a serialized value or a backend-specific
  kwarg. This matches NOTE2 exactly.
- The `CondSearchable` protocol is the only coupling point between `EntityStore` and any
  concrete adapter.
- `CChroma` (and future adapters) own everything backend-specific: serialization format,
  search kwargs, and `DocCond` mapping. No coordination across classes is needed.
- The AST dataclasses (`CompCond`, `LogicalCond`, etc.) are portable, testable in
  isolation, and rich enough to express Qdrant-level semantics.
- Operator overloading with smart flattening (`a & b & c → AND(a,b,c)`) and `~`
  with double-negation elimination are purely AST-level and backend-agnostic.

## Plan: Condition System

Add a backend-agnostic condition AST to `llm_core`, integrate it into `CChroma` via a
`CondSearchable` protocol, and thread it through `EntityStore`. Then migrate laife.

### Step 1 - Create `llm_core/vectorstores/cond.py`

New file. Contains:

- `LiteralValue = str | int | float | bool`
- Enums: `CompOp` (`EQ NE GT GTE LT LTE`), `InclusionOp` (`IN NIN`), `LogicalOp` (`AND OR`)
- Dataclasses (all inherit `_CondMixin`): `CompCond`, `InclusionCond`, `NotCond`,
  `LogicalCond`, `DocCond`
- Type alias: `AnyCond = CompCond | InclusionCond | NotCond | LogicalCond | DocCond`
- Protocol: `CondSearchable`
- Mixin: `_CondMixin`

**`_CondMixin`** provides `__and__`, `__or__`, `__invert__` with smart flattening:

```python
class _CondMixin:
    def __and__(self, other: AnyCond) -> LogicalCond:
        left  = self.children  if isinstance(self,  LogicalCond) and self.op  == LogicalOp.AND else [self]
        right = other.children if isinstance(other, LogicalCond) and other.op == LogicalOp.AND else [other]
        return LogicalCond(LogicalOp.AND, left + right)

    def __or__(self, other: AnyCond) -> LogicalCond:
        left  = self.children  if isinstance(self,  LogicalCond) and self.op  == LogicalOp.OR else [self]
        right = other.children if isinstance(other, LogicalCond) and other.op == LogicalOp.OR else [other]
        return LogicalCond(LogicalOp.OR, left + right)

    def __invert__(self) -> AnyCond:
        # double-negation elimination at the AST level
        return self.condition if isinstance(self, NotCond) else NotCond(self)
```

**`CondSearchable` protocol:**

```python
class CondSearchable(Protocol):
    def search(self, query: str, k: int, cond: AnyCond | None) -> list[Document]: ...
```

**Dataclasses** (all also inherit `_CondMixin`):

```python
@dataclass
class CompCond(_CondMixin):
    field: str
    op: CompOp
    value: LiteralValue

@dataclass
class InclusionCond(_CondMixin):
    field: str
    op: InclusionOp
    values: list[LiteralValue]

@dataclass
class NotCond(_CondMixin):
    condition: AnyCond

@dataclass
class LogicalCond(_CondMixin):
    op: LogicalOp
    children: list[AnyCond]

@dataclass
class DocCond(_CondMixin):
    text: str
    negate: bool = False
```

No Chroma imports anywhere in this file.

---

### Step 2 - Add Chroma serialization to `CChroma`

`CChroma` gains a private `_serialize(cond)` method and the public `search()` method
that satisfies `CondSearchable`.

```python
# cchroma.py

def search(self, query: str, k: int = 4, cond: AnyCond | None = None) -> list[Document]:
    meta_filter, doc_filter = self._serialize(cond)
    kwargs: dict[str, Any] = {"filter": meta_filter}
    if doc_filter is not None:
        kwargs["where_document"] = doc_filter
    return self.similarity_search(query, k=k, **kwargs)

def _serialize(self, cond: AnyCond | None) -> tuple[dict | None, dict | None]:
    """Return (metadata_filter, where_document_filter) for Chroma."""
    if cond is None:
        return None, None
    if isinstance(cond, DocCond):
        return None, self._doc_cond_to_chroma(cond)
    return self._cond_to_chroma(cond), None

def _cond_to_chroma(self, cond: AnyCond) -> dict:
    match cond:
        case CompCond(field, op, value):
            return {field: {f"${op.value}": value}}
        case InclusionCond(field, op, values):
            return {field: {f"${op.value}": values}}
        case NotCond(inner):
            return self._negate_chroma(inner)
        case LogicalCond(op, children):
            return {f"${op.value}": [self._cond_to_chroma(c) for c in children]}
        case DocCond():
            msg = "DocCond cannot be nested inside a metadata filter"
            raise TypeError(msg)

def _negate_chroma(self, cond: AnyCond) -> dict:
    """Push NotCond down to the leaf since Chroma has no top-level $not."""
    match cond:
        case CompCond(field, CompOp.EQ, value):
            return {field: {"$ne": value}}
        case CompCond(field, CompOp.NE, value):
            return {field: {"$eq": value}}
        case CompCond(field, CompOp.GT, value):
            return {field: {"$lte": value}}
        case CompCond(field, CompOp.GTE, value):
            return {field: {"$lt": value}}
        case CompCond(field, CompOp.LT, value):
            return {field: {"$gte": value}}
        case CompCond(field, CompOp.LTE, value):
            return {field: {"$gt": value}}
        case InclusionCond(field, InclusionOp.IN, values):
            return {field: {"$nin": values}}
        case InclusionCond(field, InclusionOp.NIN, values):
            return {field: {"$in": values}}
        case LogicalCond(LogicalOp.AND, children):
            # De Morgan: NOT AND → OR of negations
            return {"$or": [self._negate_chroma(c) for c in children]}
        case LogicalCond(LogicalOp.OR, children):
            # De Morgan: NOT OR → AND of negations
            return {"$and": [self._negate_chroma(c) for c in children]}
        case NotCond(inner):
            # Double negation already eliminated at AST level; but handle defensively
            return self._cond_to_chroma(inner)
        case _:
            msg = f"Cannot negate {type(cond).__name__} in Chroma serializer"
            raise TypeError(msg)

def _doc_cond_to_chroma(self, cond: DocCond) -> dict:
    op = "$not_contains" if cond.negate else "$contains"
    return {op: cond.text}
```

Note: `filter=None` must not be passed to `similarity_search` when there is no filter -
Chroma treats `filter=None` differently from omitting the argument. Use the `kwargs`
dict approach shown above.

---

### Step 3 - Update `EntityStore.search()`

Replace `**filter_kwargs` with `cond: AnyCond | None = None`. Change `self._store` type
annotation from `VectorStore` to `CondSearchable`. Delegate directly.

```python
# entity_store.py

from llm_core.vectorstores.cond import AnyCond, CondSearchable

class EntityStore:
    _store: CondSearchable  # was: VectorStore

    @overload
    def search(self, query: str, *, entity_type: None = None,
               k: int = ..., cond: AnyCond | None = ...) -> list[Document]: ...
    @overload
    def search[T: Vectorable](self, query: str, *, entity_type: type[T],
               k: int = ..., cond: AnyCond | None = ...) -> list[T]: ...

    def search(self, query: str, *, entity_type=None, k: int = 4,
               cond: AnyCond | None = None) -> Any:
        docs = self._store.search(query, k=k, cond=cond)
        if entity_type is None:
            return docs
        return [entity_type.from_document(doc) for doc in docs]
```

`VectorStoreConfig.create_store()` already returns `CChroma` (from `ChromaConfig`), and
`CChroma` now satisfies `CondSearchable`, so no config changes are needed.

---

### Step 4 - Update `__init__.py` exports

Add `AnyCond`, `CompCond`, `InclusionCond`, `NotCond`, `LogicalCond`, `DocCond`,
`CompOp`, `InclusionOp`, `LogicalOp`, and `CondSearchable` to
`llm_core/vectorstores/__init__.py`.

---

### Step 5 - Write tests for `cond.py`

New file: `tests/vectorstores/test_cond.py`. No Chroma needed; tests the pure AST.

Coverage:

- Each condition class constructs and stores its fields.
- `__and__` / `__or__` produce a flat `LogicalCond` (not nested) for chained calls.
- `a & b & c` produces `LogicalCond(AND, [a, b, c])` not `LogicalCond(AND, [LogicalCond(AND, [a, b]), c])`.
- `~cond` produces `NotCond(cond)`.
- `~~cond` returns the original `cond` (double-negation elimination).
- `CondSearchable` is a structural protocol: any class with a matching `search()` satisfies it.

---

### Step 6 - Write tests for `CChroma.search()` and `_serialize()`

Add to `tests/vectorstores/test_cchroma.py` (requires Chroma; existing tests already
handle this with `pytest.mark.skipif`).

Coverage:

- `search()` with `cond=None` returns results (smoke test, no filter).
- `search()` with a `CompCond` returns only matching documents.
- `search()` with a `LogicalCond(AND, [...])` filters correctly.
- `search()` with a `DocCond` passes `where_document` (Chroma-level filter on content).
- `search()` with a `NotCond` wrapping a `CompCond` applies the negated filter.
- `_serialize(None)` returns `(None, None)`.
- `_serialize(DocCond(...))` returns `(None, {...})` and `(None, None)` half-branches.

---

### Step 7 - Update existing tests for `EntityStore`

In `tests/vectorstores/test_entity_store.py`:

- Remove the two `filter_kwargs` pass-through tests
  (`test_search_no_entity_and_filter_kwargs`, `test_search_with_entity_and_filter_kwargs`).
- Add tests for `search(cond=CompCond("entity_type", CompOp.EQ, "type_a"))` that verify
  actual filtering behavior (not just "does not raise").

---

### Step 8 - Migrate laife

Update the migration plan doc `llm-core/scratch_space/02_core/08-migration-laife.md`, do not edit any laife code yet.

In `laife` (after `llm-core` is published or path-linked as a dev dependency):

1. Delete `src/laife/llm_services/vectorstores/cond.py` - replaced by `llm_core.vectorstores.cond`.
2. Update `tests/llm/test_cond.py` imports to use `llm_core.vectorstores.cond` types.
3. Update any import path that referenced the old module (grep `from laife.llm_services.vectorstores.cond`).
4. Include the `cond.py` → `llm_core.vectorstores.cond` mapping in the import substitution table.

---

### Notes

1. `filter=None` vs omitting `filter` - Chroma exhibits different behavior when
   `filter=None` is passed explicitly vs the kwarg being absent. The `kwargs` dict
   approach in Step 2 avoids the issue cleanly.

2. `DocCond` inside `LogicalCond` is disallowed in the Chroma serializer (raises
   `TypeError`). Chroma requires `where_document` to be a separate argument and cannot
   be composed with `where`. This restriction is Chroma-specific; a Qdrant serializer
   may allow it. Document this clearly in the method docstring.

3. The `CondSearchable` protocol uses a positional `k` parameter, which conflicts with
   the existing `Chroma.similarity_search(query, k=4, **kwargs)` signature. The `search`
   method added to `CChroma` should match the protocol signature exactly, and pyright
   will validate structural conformance.

4. `VectorStoreConfig.create_store()` return type is `VectorStore` (base class). Pyright
   will see `EntityStore._store` as `CondSearchable`. Since `ChromaConfig.create_store()`
   returns `CChroma` (which satisfies the protocol), no runtime issue exists - but a
   pyright override annotation or a narrower return type on `create_store()` may be
   needed to satisfy the type checker.

5. Parallelizable work: Steps 1 and 5 (AST + pure unit tests) can be written
   independently of Steps 2, 6 (Chroma integration). Steps 3 and 4 depend on Step 1.
   Step 7 depends on Step 3. Step 8 depends on all preceding steps.
