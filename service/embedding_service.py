from __future__ import annotations

import hashlib
import math
from typing import List, Protocol


class EmbeddingProvider(Protocol):
    embedding_dim: int
    model_name: str

    def embed_text(self, text: str) -> List[float]:
        ...


class HashEmbeddingProvider:
    """
    Minimal local embedding provider.

    It is not a semantic embedding model. It exists so the project can run without
    an external API or a local ML model. Replace this provider with a real embedding
    provider later, but keep embedding_dim consistent with the pgvector column.
    """

    embedding_dim = 384
    model_name = "local-hash-embedding-384"

    def embed_text(self, text: str) -> List[float]:
        text = (text or "").strip()
        vector = [0.0] * self.embedding_dim

        if not text:
            return vector

        tokens = self._tokenize(text)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        return self._l2_normalize(vector)

    def _tokenize(self, text: str) -> List[str]:
        # Works acceptably for mixed Chinese/English in development:
        # - English is split by whitespace
        # - Chinese text also gets character-level fallback tokens
        rough_tokens = text.lower().replace("，", " ").replace("。", " ").split()
        char_tokens = [ch for ch in text if not ch.isspace()]
        return rough_tokens + char_tokens

    def _l2_normalize(self, vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self.provider = provider or HashEmbeddingProvider()

    @property
    def embedding_dim(self) -> int:
        return self.provider.embedding_dim

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def embed_memory(self, title: str, content: str, memory_type: str | None = None) -> List[float]:
        text = self._build_memory_text(title=title, content=content, memory_type=memory_type)
        return self.provider.embed_text(text)

    def embed_query(self, query: str) -> List[float]:
        return self.provider.embed_text(query)

    def _build_memory_text(self, title: str, content: str, memory_type: str | None = None) -> str:
        return f"类型：{memory_type or 'unknown'}\n标题：{title or ''}\n内容：{content or ''}"
