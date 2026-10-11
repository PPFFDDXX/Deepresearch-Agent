"""Transport-independent production research orchestration."""
from collections.abc import Callable, Sequence
from pathlib import Path

from app.config import Settings
from app.evaluation import evaluate_state
from app.graph import run_research
from app.llm import ChatCompletionsModel
from app.state import AgentState
from app.research_context import build_research_context
from app.tools.models import ToolError, WebSearchResponse
from app.tools.web_search import TavilySearchBackend, build_web_search


def build_live_search(settings: Settings, *, enabled: bool = True) -> Callable[[str], WebSearchResponse]:
    def search(query: str) -> WebSearchResponse:
        if not enabled:
            raise ToolError("configuration", "Web search is disabled for local-only research")
        return build_web_search(TavilySearchBackend(settings), settings)(query)
    return search


def serialize_state(state: AgentState) -> dict:
    return {key: ([item.model_dump(mode="json") for item in value] if isinstance(value, list)
                  else value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
            for key, value in state.items()}


def execute_research(query: str, settings: Settings, *, documents: Sequence[Path] = (),
                     local_only: bool = False, include_archived: bool = False) -> dict:
    if local_only and not documents:
        raise ValueError("Local-only research requires documents")
    rag = None
    if documents:
        from app.rag.pipeline import build_local_rag
        rag = build_local_rag(documents, settings, include_archived=include_archived)
    state = run_research(query, ChatCompletionsModel(settings), settings,
        research_context=build_research_context(rag.documents if rag else (), local_only=local_only),
        web_search=build_live_search(settings, enabled=not local_only),
        retrieve_documents=rag.retrieve_documents if rag else None,
        local_document_context=rag.describe_documents() if rag else "", web_search_available=not local_only,
        writer_model=ChatCompletionsModel(settings, timeout_seconds=settings.writer_timeout_seconds))
    return {**serialize_state(state), "evaluation": evaluate_state(state)}
