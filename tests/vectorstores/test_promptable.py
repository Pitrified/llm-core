"""Tests for the Promptable protocol."""

from llm_core.vectorstores.promptable import Promptable


class DummyPromptable:
    """Dummy class satisfying Promptable."""

    def to_prompt(self) -> str:  # noqa: D102
        return "I am promptable"


class NotPromptable:
    """Class that does NOT satisfy Promptable."""


class TestPromptableProtocol:
    """Tests for Promptable protocol conformance."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        """A class with to_prompt satisfies Promptable."""
        obj = DummyPromptable()
        assert isinstance(obj, Promptable)

    def test_non_conforming_class_fails(self) -> None:
        """A class without to_prompt does not satisfy Promptable."""
        obj = NotPromptable()
        assert not isinstance(obj, Promptable)

    def test_to_prompt_returns_string(self) -> None:
        """to_prompt returns the expected string."""
        obj = DummyPromptable()
        assert obj.to_prompt() == "I am promptable"
