"""Tests for the backend-agnostic condition AST."""

from langchain_core.documents import Document

from llm_core.vectorstores.cond import AnyCond
from llm_core.vectorstores.cond import CompCond
from llm_core.vectorstores.cond import CompOp
from llm_core.vectorstores.cond import CondSearchable
from llm_core.vectorstores.cond import DocCond
from llm_core.vectorstores.cond import InclusionCond
from llm_core.vectorstores.cond import InclusionOp
from llm_core.vectorstores.cond import LogicalCond
from llm_core.vectorstores.cond import LogicalOp
from llm_core.vectorstores.cond import NotCond


class TestCompCond:
    """Tests for CompCond construction."""

    def test_stores_fields(self) -> None:
        """CompCond stores field, op, and value."""
        c = CompCond("age", CompOp.GT, 30)
        assert c.field == "age"
        assert c.op == CompOp.GT
        assert c.value == 30


class TestInclusionCond:
    """Tests for InclusionCond construction."""

    def test_stores_fields(self) -> None:
        """InclusionCond stores field, op, and values."""
        c = InclusionCond("status", InclusionOp.IN, ["active", "pending"])
        assert c.field == "status"
        assert c.op == InclusionOp.IN
        assert c.values == ["active", "pending"]


class TestNotCond:
    """Tests for NotCond construction."""

    def test_wraps_condition(self) -> None:
        """NotCond wraps another condition."""
        inner = CompCond("age", CompOp.GT, 30)
        n = NotCond(inner)
        assert n.condition is inner


class TestLogicalCond:
    """Tests for LogicalCond construction."""

    def test_stores_op_and_children(self) -> None:
        """LogicalCond stores operator and children list."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        lc = LogicalCond(LogicalOp.AND, [a, b])
        assert lc.op == LogicalOp.AND
        assert lc.children == [a, b]

    def test_single_child_allowed(self) -> None:
        """A single-child LogicalCond is valid."""
        a = CompCond("a", CompOp.EQ, 1)
        lc = LogicalCond(LogicalOp.AND, [a])
        assert len(lc.children) == 1


class TestDocCond:
    """Tests for DocCond construction."""

    def test_default_negate_false(self) -> None:
        """DocCond defaults negate to False."""
        d = DocCond("hello")
        assert d.text == "hello"
        assert d.negate is False

    def test_negate_true(self) -> None:
        """DocCond can be created with negate=True."""
        d = DocCond("hello", negate=True)
        assert d.negate is True


class TestOperatorOverloadingAnd:
    """Tests for __and__ with smart flattening."""

    def test_two_comp_conds(self) -> None:
        """A & b produces AND([a, b])."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        result = a & b
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.AND
        assert result.children == [a, b]

    def test_three_comp_conds_flat(self) -> None:
        """A & b & c produces AND([a, b, c]), not AND(AND(a,b), c)."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        c = CompCond("c", CompOp.EQ, 3)
        result = a & b & c
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.AND
        assert result.children == [a, b, c]

    def test_four_comp_conds_flat(self) -> None:
        """A & b & c & d produces AND([a, b, c, d])."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        c = CompCond("c", CompOp.EQ, 3)
        d = CompCond("d", CompOp.EQ, 4)
        result = a & b & c & d
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.AND
        assert result.children == [a, b, c, d]

    def test_and_does_not_flatten_or(self) -> None:
        """(a | b) & c produces AND([OR(a,b), c]), not AND(a,b,c)."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        c = CompCond("c", CompOp.EQ, 3)
        or_cond = a | b
        result = or_cond & c
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.AND
        assert result.children == [or_cond, c]


class TestOperatorOverloadingOr:
    """Tests for __or__ with smart flattening."""

    def test_two_comp_conds(self) -> None:
        """A | b produces OR([a, b])."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        result = a | b
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.OR
        assert result.children == [a, b]

    def test_three_comp_conds_flat(self) -> None:
        """A | b | c produces OR([a, b, c])."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        c = CompCond("c", CompOp.EQ, 3)
        result = a | b | c
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.OR
        assert result.children == [a, b, c]

    def test_or_does_not_flatten_and(self) -> None:
        """(a & b) | c produces OR([AND(a,b), c])."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        c = CompCond("c", CompOp.EQ, 3)
        and_cond = a & b
        result = and_cond | c
        assert isinstance(result, LogicalCond)
        assert result.op == LogicalOp.OR
        assert result.children == [and_cond, c]


class TestOperatorOverloadingInvert:
    """Tests for __invert__ (NOT) and double-negation elimination."""

    def test_invert_comp_cond(self) -> None:
        """~CompCond produces NotCond."""
        c = CompCond("a", CompOp.EQ, 1)
        result = ~c
        assert isinstance(result, NotCond)
        assert result.condition is c

    def test_double_negation_eliminated(self) -> None:
        """~~cond returns the original condition."""
        c = CompCond("a", CompOp.EQ, 1)
        result = ~~c
        assert result is c

    def test_invert_logical_cond(self) -> None:
        """~LogicalCond produces NotCond wrapping the logical."""
        a = CompCond("a", CompOp.EQ, 1)
        b = CompCond("b", CompOp.EQ, 2)
        lc = a & b
        result = ~lc
        assert isinstance(result, NotCond)
        assert result.condition is lc

    def test_invert_inclusion_cond(self) -> None:
        """~InclusionCond produces NotCond."""
        c = InclusionCond("status", InclusionOp.IN, ["a", "b"])
        result = ~c
        assert isinstance(result, NotCond)
        assert result.condition is c

    def test_invert_doc_cond(self) -> None:
        """~DocCond produces NotCond wrapping the DocCond."""
        c = DocCond("hello")
        result = ~c
        assert isinstance(result, NotCond)
        assert result.condition is c


class TestCondSearchableProtocol:
    """Tests for the CondSearchable structural protocol."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        """A class with a matching cond_search() satisfies CondSearchable."""

        class FakeStore:
            def cond_search(
                self,
                query: str,  # noqa: ARG002
                k: int,  # noqa: ARG002
                cond: AnyCond | None,  # noqa: ARG002
            ) -> list[Document]:
                return []

        assert isinstance(FakeStore(), CondSearchable)

    def test_non_conforming_class_fails(self) -> None:
        """A class without cond_search() does not satisfy CondSearchable."""

        class NotAStore:
            pass

        assert not isinstance(NotAStore(), CondSearchable)
