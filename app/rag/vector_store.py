"""In-memory FAISS cosine retrieval over immutable chunk metadata."""

from collections.abc import Sequence

import faiss
import numpy as np

from app.observability import span

from app.rag.embeddings import EmbeddingModel
from app.rag.models import ChunkHit, DocumentChunk, RagError


def normalized_matrix(vectors: Sequence[Sequence[float]], rows: int, dimensions: int | None = None) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise RagError("Embedding vectors must be finite, rectangular and match the input count")
    if dimensions is not None and matrix.shape[1] != dimensions:
        raise RagError("Query embedding dimensions do not match document embeddings")
    matrix = np.ascontiguousarray(matrix)
    faiss.normalize_L2(matrix)
    return matrix


class FaissVectorStore:
    def __init__(self, chunks: Sequence[DocumentChunk], embeddings: EmbeddingModel):
        if not chunks:
            raise RagError("Cannot build an index without document chunks")
        self.chunks = tuple(chunks)
        self.embeddings = embeddings
        try:
            with span("embedding.documents", "embedding") as attributes:
                attributes["input_count"] = len(chunks)
                matrix = normalized_matrix(embeddings.embed([item.text for item in chunks]), len(chunks))
                attributes["dimensions"] = matrix.shape[1]
        except (ValueError, TypeError):
            raise RagError("Invalid document embedding response") from None
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def search(self, query: str, *, top_k: int, min_score: float) -> list[ChunkHit]:
        if not isinstance(query, str) or not query.strip():
            raise RagError("Retrieval query must be a nonempty string")
        if top_k < 1 or not 0 <= min_score <= 1:
            raise RagError("Invalid retrieval limit or score threshold")
        try:
            with span("embedding.query", "embedding") as attributes:
                attributes["input_count"] = 1
                matrix = normalized_matrix(self.embeddings.embed([query]), 1, self.index.d)
                attributes["dimensions"] = matrix.shape[1]
        except (ValueError, TypeError):
            raise RagError("Invalid query embedding response") from None
        if not np.any(matrix):
            return []
        with span("faiss.search", "rag"):
            scores, indices = self.index.search(matrix, min(top_k, len(self.chunks)))
        return [ChunkHit(chunk=self.chunks[int(index)], score=float(np.clip(score, -1, 1)))
                for score, index in zip(scores[0], indices[0]) if index >= 0 and score >= min_score]
