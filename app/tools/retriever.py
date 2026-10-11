"""retrieve_documents(query) adapter, independent of AgentState and LLM decisions."""

from collections.abc import Callable

from app.rag.models import RagError
from app.rag.vector_store import FaissVectorStore
from app.tools.models import ResearchMaterial, ToolError


def build_retriever(store: FaissVectorStore, *, top_k: int, min_score: float) -> Callable[[str], list[ResearchMaterial]]:
    def retrieve_documents(query: str) -> list[ResearchMaterial]:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("invalid_input", "Retrieval query must be a nonempty string")
        try:
            hits = store.search(query, top_k=top_k, min_score=min_score)
        except RagError as exc:
            raise ToolError("invalid_response", str(exc)) from None
        return [ResearchMaterial(title=hit.chunk.title, locator=hit.chunk.locator, source_type="local",
                                 text=hit.chunk.text, location=hit.chunk.location, origin="local_chunk")
                for hit in hits]
    return retrieve_documents
