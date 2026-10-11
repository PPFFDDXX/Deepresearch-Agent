"""Replaceable embeddings: offline lexical hashing or compatible /embeddings API."""

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings
from app.observability import span
from app.rag.models import RagError


class EmbeddingModel(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, preserving input order."""
        ...


class HashingEmbeddings:
    """Deterministic lexical feature vectors, not pretrained semantic embeddings."""

    def __init__(self, dimensions: int = 1024):
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            tokens = []
            for match in re.finditer(r"[a-z0-9]+|[\u3400-\u9fff]+", text.casefold()):
                word = match.group()
                if word.isascii():
                    tokens.append("word:" + word)
                else:
                    for size in (1, 2, 3):
                        tokens.extend("cjk:" + word[i:i + size] for i in range(max(0, len(word) - size + 1)))
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token, count in Counter(tokens).items():
                digest = hashlib.sha256(token.encode()).digest()
                position = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1 if digest[8] % 2 else -1
                vector[position] += sign * (1 + np.log(count))
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors


class ApiEmbeddings:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        if settings.embedding_api_key is None or not settings.embedding_api_key.get_secret_value().strip():
            raise RagError("Configure DR_EMBEDDING_API_KEY for the API embedding provider")
        self.settings = settings
        self.transport = transport
        is_bge_m3 = settings.embedding_model.casefold() in ("baai/bge-m3", "pro/baai/bge-m3")
        if is_bge_m3 and settings.embedding_expected_dimensions not in (None, 1024):
            raise RagError("BGE-M3 dense embeddings require 1024 dimensions")
        self.dimensions = 1024 if is_bge_m3 else settings.embedding_expected_dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise RagError("Embedding input must be a sequence of nonempty text strings")
        output = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = list(texts[start:start + self.settings.embedding_batch_size])
            output.extend(self._embed_batch(batch))
        return output

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        settings = self.settings
        last_error = "Unknown embedding failure"
        for attempt in range(1, settings.embedding_max_attempts + 1):
            try:
                with span("embedding.request", "embedding") as attributes, httpx.Client(
                    timeout=settings.embedding_timeout_seconds, transport=self.transport,
                ) as client:
                    attributes.update(attempt=attempt, input_count=len(texts))
                    response = client.post(settings.embedding_base_url.rstrip("/") + "/embeddings",
                        headers={"Authorization": f"Bearer {settings.embedding_api_key.get_secret_value()}"},
                        json={"model": settings.embedding_model, "input": texts, "encoding_format": "float"})
                    attributes["http_status"] = response.status_code
                    attributes["exit_code"] = int(response.is_error)
                if response.is_error:
                    if response.status_code not in (408, 429) and response.status_code < 500:
                        raise RagError(f"Embedding API returned HTTP {response.status_code}")
                    last_error = f"Embedding API returned HTTP {response.status_code}"
                    continue
                data = response.json()["data"]
                if not isinstance(data, list) or sorted(item["index"] for item in data) != list(range(len(texts))):
                    raise ValueError("Invalid embedding response indices")
                ordered = sorted(data, key=lambda item: item["index"])
                matrix = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
                if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                    raise ValueError("Invalid embedding vectors")
                if np.any(np.linalg.norm(matrix, axis=1) == 0):
                    raise ValueError("Embedding API returned zero vectors")
                actual_dimensions = matrix.shape[1]
                if self.dimensions is not None and actual_dimensions != self.dimensions:
                    raise RagError(f"Embedding dimension mismatch: expected {self.dimensions}, received {actual_dimensions}")
                self.dimensions = actual_dimensions
                return matrix.tolist()
            except (httpx.RequestError, ValueError, KeyError, TypeError):
                last_error = "Embedding request failed or returned invalid vectors"
        raise RagError(f"{last_error} (attempts={settings.embedding_max_attempts})")


def build_embeddings(settings: Settings) -> EmbeddingModel:
    return (HashingEmbeddings(settings.embedding_dimensions) if settings.embedding_provider == "hashing"
            else ApiEmbeddings(settings))
