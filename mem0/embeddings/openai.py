import os
import warnings
from typing import Literal, Optional, List

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

def _ensure_dimensional(vec: List[float], target_dim: int = 1536) -> List[float]:
    length = len(vec)
    if length > target_dim:
        # 超长则截断
        print(
            f"The current vector dimension is {length}, "
            f"and the vector dimension cannot exceed {target_dim}. "
            f"The first {target_dim} dimensions are automatically captured"
        )
        return vec[:target_dim]
    # 不足则补零
    return vec + [0.0] * (target_dim - length)

class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        # resp = self.client.embeddings.create(
        #     input=[text],
        #     model=self.config.model,
        #     dimensions=self.config.embedding_dims
        # )
        resp = self.client.embeddings.create(
            input=[text],
            model=self.config.model
        )
        raw_vec = resp.data[0].embedding  # 可能是 1024 也可能是 1536
        # 强制调整到目标维度并返回
        return _ensure_dimensional(raw_vec, self.config.embedding_dims)
