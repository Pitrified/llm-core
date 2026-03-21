"""Lightweight protocol for the ``to_prompt()`` convention.

Any domain object that feeds into an LLM context can implement this protocol
to signal that it can render itself as a prompt-ready string.

Example:
    ::

        class Player:
            name: str
            health: int

            def to_prompt(self) -> str:
                return f"{self.name} (HP: {self.health})"
"""

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class Promptable(Protocol):
    """Structural protocol for objects that render as prompt text."""

    def to_prompt(self) -> str:
        """Return a prompt-ready string representation of the object."""
        ...
