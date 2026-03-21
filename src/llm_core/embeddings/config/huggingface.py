"""HuggingFace / SentenceTransformer embeddings configuration."""

from llm_core.embeddings.config.base import EmbeddingsConfig


class HuggingFaceEmbeddingsConfig(EmbeddingsConfig):
    """Configuration for HuggingFace / SentenceTransformer embedding models.

    Attributes:
        model: SentenceTransformer model ID. Defaults to
            "sentence-transformers/all-mpnet-base-v2".
        provider: Provider name for init_embeddings. Always "huggingface".
    """

    model: str = "sentence-transformers/all-mpnet-base-v2"
    provider: str = "huggingface"
