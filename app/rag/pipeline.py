"""Ingestion composition: documents -> chunks -> embeddings -> FAISS -> tool."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.observability import span
from app.rag.chunker import chunk_document
from app.rag.embeddings import EmbeddingModel, build_embeddings
from app.rag.loader import load_documents
from app.rag.models import LocalDocument, RagError
from app.rag.vector_store import FaissVectorStore
from app.tools.models import ResearchMaterial
from app.tools.retriever import build_retriever


@dataclass(frozen=True)
class LocalRag:
    documents: tuple[LocalDocument, ...]
    store: FaissVectorStore
    retrieve_documents: Callable[[str], list[ResearchMaterial]]
    excluded_archived: int

    def describe_documents(self) -> str:
        """Provide a bounded inventory, not raw document instructions, to planning."""
        return "Local document inventory:\n" + "\n".join(
            f"- {Path(item.locator).name}: {item.title}; version={item.version or 'unspecified'}; "
            f"effective={item.effective_date or 'unspecified'}; archived={item.archived}"
            for item in self.documents
        ) + f"\nArchived documents excluded: {self.excluded_archived}."


def build_local_rag(
    paths: Sequence[Path], settings: Settings, *, include_archived: bool = False,
    embeddings: EmbeddingModel | None = None,
) -> LocalRag:
    with span("rag.load_documents", "rag") as attributes:
        loaded = load_documents(paths, settings)
        attributes["document_count"] = len(loaded)
    documents = [item for item in loaded if include_archived or not item.archived]
    if not documents:
        raise RagError("No active documents available; use include_archived for historical retrieval")
    with span("rag.chunk_documents", "rag") as attributes:
        chunks = [chunk for item in documents for chunk in chunk_document(item, settings.rag_chunk_size, settings.rag_chunk_overlap)]
        attributes["chunk_count"] = len(chunks)
    with span("rag.build_index", "rag") as attributes:
        store = FaissVectorStore(chunks, embeddings if embeddings is not None else build_embeddings(settings))
        attributes.update(chunk_count=len(chunks), dimensions=store.index.d)
    retriever = build_retriever(store, top_k=min(settings.rag_top_k, settings.search_top_k), min_score=settings.rag_min_score)
    return LocalRag(tuple(documents), store, retriever, len(loaded) - len(documents))
