"""Backend-agnostic condition AST for filtering vector store searches.

Conditions are pure data (dataclasses) with no backend-specific concepts.
Each concrete vector store adapter serializes the AST into its native filter
format. Operator overloading allows composing conditions with ``&``, ``|``,
and ``~``.

Example:
    ::

        from llm_core.vectorstores.cond import CompCond, CompOp, LogicalOp

        age_filter = CompCond("age", CompOp.GT, 30)
        name_filter = CompCond("name", CompOp.EQ, "Alice")
        combined = age_filter & name_filter   # LogicalCond(AND, [age, name])
        negated = ~age_filter                 # NotCond(age_filter)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from langchain_core.documents import Document

LiteralValue = str | int | float | bool
"""Scalar types allowed in condition values."""


class CompOp(Enum):
    """Comparison operators for field-level conditions."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class InclusionOp(Enum):
    """Inclusion/exclusion operators for set membership conditions."""

    IN = "in"
    NIN = "nin"


class LogicalOp(Enum):
    """Logical combinators for composing conditions."""

    AND = "and"
    OR = "or"


class _CondMixin:
    """Mixin providing ``&``, ``|``, ``~`` operators for condition composition.

    Smart flattening ensures ``a & b & c`` produces ``AND(a, b, c)``
    instead of ``AND(AND(a, b), c)``.
    """

    def __and__(self, other: AnyCond) -> LogicalCond:
        """Combine with AND, flattening nested ANDs."""
        left: list[AnyCond] = (
            self.children  # type: ignore[attr-defined]
            if isinstance(self, LogicalCond) and self.op == LogicalOp.AND
            else [self]  # type: ignore[list-item]
        )
        right: list[AnyCond] = (
            other.children
            if isinstance(other, LogicalCond) and other.op == LogicalOp.AND
            else [other]
        )
        return LogicalCond(LogicalOp.AND, [*left, *right])

    def __or__(self, other: AnyCond) -> LogicalCond:
        """Combine with OR, flattening nested ORs."""
        left: list[AnyCond] = (
            self.children  # type: ignore[attr-defined]
            if isinstance(self, LogicalCond) and self.op == LogicalOp.OR
            else [self]  # type: ignore[list-item]
        )
        right: list[AnyCond] = (
            other.children
            if isinstance(other, LogicalCond) and other.op == LogicalOp.OR
            else [other]
        )
        return LogicalCond(LogicalOp.OR, [*left, *right])

    def __invert__(self) -> AnyCond:
        """Negate the condition; double-negation is eliminated at the AST level."""
        if isinstance(self, NotCond):
            return self.condition
        return NotCond(self)  # type: ignore[arg-type]


@dataclass
class CompCond(_CondMixin):
    """Field-level comparison condition (e.g. ``age > 30``)."""

    field: str
    op: CompOp
    value: LiteralValue


@dataclass
class InclusionCond(_CondMixin):
    """Field-level set membership condition (e.g. ``status IN [a, b]``)."""

    field: str
    op: InclusionOp
    values: list[LiteralValue]


@dataclass
class NotCond(_CondMixin):
    """Logical negation wrapping any condition."""

    condition: AnyCond


@dataclass
class LogicalCond(_CondMixin):
    """Logical combination (AND/OR) of child conditions."""

    op: LogicalOp
    children: list[AnyCond]


@dataclass
class DocCond(_CondMixin):
    """Document-level text filter (e.g. contains / not_contains).

    Maps to ``where_document`` in Chroma and ``FullTextMatch`` in Qdrant.
    """

    text: str
    negate: bool = False


AnyCond = CompCond | InclusionCond | NotCond | LogicalCond | DocCond
"""Union of all condition node types."""


@runtime_checkable
class CondSearchable(Protocol):
    """Protocol that concrete vector store adapters must satisfy.

    ``EntityStore`` delegates search calls to this protocol, remaining
    completely unaware of backend-specific serialization.
    """

    def cond_search(
        self,
        query: str,
        k: int,
        cond: AnyCond | None,
    ) -> list[Document]:
        """Search with an optional condition filter.

        Args:
            query: Free-text similarity query.
            k: Maximum number of results.
            cond: Optional condition tree to filter results.

        Returns:
            Up to *k* matching documents.
        """
        ...
