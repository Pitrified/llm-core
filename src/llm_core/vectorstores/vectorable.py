"""Protocol for entities that can be serialized to/from a vector-store Document.

``Vectorable`` is a structural ``@runtime_checkable`` protocol - no
inheritance required. Any class that implements ``to_document`` and
``from_document`` satisfies it.

Example:
    ::

        class MyEntity:
            def to_document(self) -> Document:
                return Document(
                    page_content=self.text,
                    metadata={"entity_type": "my_entity"},
                )

            @classmethod
            def from_document(cls, doc: Document) -> Self:
                return cls(text=doc.page_content)
"""

from typing import Protocol
from typing import Self
from typing import runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Vectorable(Protocol):
    """Structural protocol for entities that round-trip through a Document.

    Implementors must:
    - Write ``entity_type`` into ``doc.metadata`` inside ``to_document``.
    - Have a pure ``from_document`` that requires only the ``Document``
      (no vector-store reference).
    """

    def to_document(self) -> Document:
        """Serialize the entity to a LangChain Document."""
        ...

    @classmethod
    def from_document(cls, doc: Document) -> Self:
        """Reconstruct an entity from a LangChain Document."""
        ...
