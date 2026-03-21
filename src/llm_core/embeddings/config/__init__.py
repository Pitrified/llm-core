"""Embeddings config submodule."""

from llm_core.embeddings.config.azure_openai import AzureOpenAIEmbeddingsConfig
from llm_core.embeddings.config.base import EmbeddingsConfig
from llm_core.embeddings.config.huggingface import HuggingFaceEmbeddingsConfig
from llm_core.embeddings.config.ollama import OllamaEmbeddingsConfig
from llm_core.embeddings.config.openai import OpenAIEmbeddingsConfig

__all__ = [
    "AzureOpenAIEmbeddingsConfig",
    "EmbeddingsConfig",
    "HuggingFaceEmbeddingsConfig",
    "OllamaEmbeddingsConfig",
    "OpenAIEmbeddingsConfig",
]
