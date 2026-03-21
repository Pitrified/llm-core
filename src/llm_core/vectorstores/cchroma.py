"""Dedup-aware Chroma vector store with condition-based filtering.

``CChroma`` combines ``DeduplicatingMixin`` with Chroma and implements the
``CondSearchable`` protocol. It owns all Chroma-specific serialization of the
backend-agnostic condition AST.

Example:
    ::

        from llm_core.vectorstores.cchroma import CChroma
        from llm_core.vectorstores.cond import CompCond, CompOp

        store = CChroma(collection_name="docs")
        store.add_documents([doc1, doc2])  # duplicates are skipped
        results = store.cond_search("hello", k=5, cond=CompCond("tag", CompOp.EQ, "x"))
"""

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from llm_core.vectorstores.cond import AnyCond
from llm_core.vectorstores.cond import CompCond
from llm_core.vectorstores.cond import CompOp
from llm_core.vectorstores.cond import DocCond
from llm_core.vectorstores.cond import InclusionCond
from llm_core.vectorstores.cond import InclusionOp
from llm_core.vectorstores.cond import LogicalCond
from llm_core.vectorstores.cond import LogicalOp
from llm_core.vectorstores.cond import NotCond
from llm_core.vectorstores.mixins.deduplicating import DeduplicatingMixin

_COMP_NEGATE: dict[CompOp, CompOp] = {
    CompOp.EQ: CompOp.NE,
    CompOp.NE: CompOp.EQ,
    CompOp.GT: CompOp.LTE,
    CompOp.GTE: CompOp.LT,
    CompOp.LT: CompOp.GTE,
    CompOp.LTE: CompOp.GT,
}

_INCLUSION_NEGATE: dict[InclusionOp, InclusionOp] = {
    InclusionOp.IN: InclusionOp.NIN,
    InclusionOp.NIN: InclusionOp.IN,
}

_CHROMA_OP: dict[CompOp, str] = {
    CompOp.EQ: "$eq",
    CompOp.NE: "$ne",
    CompOp.GT: "$gt",
    CompOp.GTE: "$gte",
    CompOp.LT: "$lt",
    CompOp.LTE: "$lte",
}

_CHROMA_INCLUSION_OP: dict[InclusionOp, str] = {
    InclusionOp.IN: "$in",
    InclusionOp.NIN: "$nin",
}


class CChroma(DeduplicatingMixin, Chroma):
    """Dedup-aware Chroma vector store satisfying ``CondSearchable``.

    MRO: CChroma -> DeduplicatingMixin -> Chroma -> VectorStore

    ``add_documents()`` from the mixin intercepts each call, hashes every
    document via SHA-256, filters out duplicates, then forwards only new
    documents to ``Chroma.add_documents()`` via ``super()``.

    ``cond_search()`` serializes the backend-agnostic condition AST into
    Chroma's ``filter`` / ``where_document`` kwargs and delegates to
    ``similarity_search()``.
    """

    # -- CondSearchable protocol ---------------------------------------------

    def cond_search(
        self,
        query: str,
        k: int = 4,
        cond: AnyCond | None = None,
    ) -> list[Document]:
        """Similarity search with optional condition filtering.

        Args:
            query: Free-text similarity query.
            k: Maximum number of results.
            cond: Optional condition tree to filter results.

        Returns:
            Up to *k* matching documents.
        """
        meta_filter, doc_filter = self._serialize(cond)
        kwargs: dict[str, Any] = {}
        if meta_filter is not None:
            kwargs["filter"] = meta_filter
        if doc_filter is not None:
            kwargs["where_document"] = doc_filter
        return self.similarity_search(query, k=k, **kwargs)

    # -- private serialization -----------------------------------------------

    def _serialize(
        self,
        cond: AnyCond | None,
    ) -> tuple[dict | None, dict | None]:
        """Serialize a condition into Chroma's filter kwargs.

        Returns:
            A ``(metadata_filter, where_document_filter)`` tuple. Either or
            both may be ``None``.
        """
        if cond is None:
            return None, None
        if isinstance(cond, DocCond):
            return None, self._doc_cond_to_chroma(cond)
        return self._cond_to_chroma(cond), None

    def _cond_to_chroma(self, cond: AnyCond) -> dict:
        """Recursively convert a metadata condition to a Chroma filter dict."""
        match cond:
            case CompCond(field, op, value):
                return {field: {_CHROMA_OP[op]: value}}
            case InclusionCond(field, op, values):
                return {field: {_CHROMA_INCLUSION_OP[op]: values}}
            case NotCond(inner):
                return self._negate_chroma(inner)
            case LogicalCond(op, children):
                chroma_op = "$and" if op == LogicalOp.AND else "$or"
                return {chroma_op: [self._cond_to_chroma(c) for c in children]}
            case DocCond():
                msg = "DocCond cannot be nested inside a metadata filter"
                raise TypeError(msg)
            case _:  # pragma: no cover
                msg = f"Unknown condition type: {type(cond).__name__}"
                raise TypeError(msg)

    def _negate_chroma(self, cond: AnyCond) -> dict:
        """Push negation down to leaf nodes (Chroma has no top-level ``$not``)."""
        match cond:
            case CompCond(field, op, value):
                return {field: {_CHROMA_OP[_COMP_NEGATE[op]]: value}}
            case InclusionCond(field, op, values):
                return {field: {_CHROMA_INCLUSION_OP[_INCLUSION_NEGATE[op]]: values}}
            case LogicalCond(LogicalOp.AND, children):
                # De Morgan: NOT AND -> OR of negations
                return {"$or": [self._negate_chroma(c) for c in children]}
            case LogicalCond(LogicalOp.OR, children):
                # De Morgan: NOT OR -> AND of negations
                return {"$and": [self._negate_chroma(c) for c in children]}
            case NotCond(inner):
                return self._cond_to_chroma(inner)
            case DocCond():
                msg = "Cannot negate DocCond inside a metadata filter"
                raise TypeError(msg)
            case _:  # pragma: no cover
                msg = f"Cannot negate {type(cond).__name__} in Chroma serializer"
                raise TypeError(msg)

    @staticmethod
    def _doc_cond_to_chroma(cond: DocCond) -> dict:
        """Convert a ``DocCond`` to Chroma's ``where_document`` dict."""
        op = "$not_contains" if cond.negate else "$contains"
        return {op: cond.text}
