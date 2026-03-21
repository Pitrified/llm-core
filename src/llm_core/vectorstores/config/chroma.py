"""Chroma vector store configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_core.vectorstores.config.base import VectorStoreConfig

if TYPE_CHECKING:
    from llm_core.vectorstores.cchroma import CChroma


class ChromaConfig(VectorStoreConfig):
    """Config for a local or server-backed Chroma vector store.

    Three deployment modes are supported - pick the fields that apply:

    * **In-memory / ephemeral** - leave all optional fields as ``None``.
    * **Local persistent** - set ``persist_directory``.
    * **Remote server** - set ``host`` and optionally ``port`` / ``ssl``.

    Attributes:
        persist_directory: Filesystem path for a persistent local Chroma store.
        host: Hostname of a deployed Chroma server.
        port: Port of a deployed Chroma server (Chroma default: 8000).
        ssl: Whether to use SSL when connecting to a remote Chroma server.
    """

    persist_directory: str | None = None
    host: str | None = None
    port: int | None = None
    ssl: bool = False

    def create_store(self) -> CChroma:
        """Return a ``CChroma`` instance constructed from this config.

        Returns:
            A dedup-aware Chroma vector store.
        """
        from llm_core.vectorstores.cchroma import CChroma  # noqa: PLC0415

        embedding_function = None
        if self.embeddings_config is not None:
            embedding_function = self.embeddings_config.create_embeddings()

        return CChroma(
            collection_name=self.collection_name,
            embedding_function=embedding_function,
            persist_directory=self.persist_directory,
        )
