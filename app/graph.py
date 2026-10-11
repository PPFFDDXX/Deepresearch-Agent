"""Plan, research a batch, evaluate, and refine within a bounded retry budget."""

from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.planner import build_planner_node
from app.agents.critic import build_critic_node
from app.agents.researcher import build_researcher_node
from app.agents.writer import build_writer_node
from app.config import Settings
from app.llm import StructuredModel
from app.observability import current_recorder, observe_model, span, trace_run, traced_callable
from app.state import AgentState, create_initial_state, validate_state
from app.tools.models import ResearchMaterial, WebSearchResponse
from app.research_context import ResearchContext


def route_after_research(state: AgentState) -> Literal["researcher", "critic"]:
    """Route using the merged state, after the Researcher advances current_task."""
    validate_state(state)
    return "researcher" if state["current_task"] is not None else "critic"


def route_after_critic(state: AgentState) -> Literal["researcher", "writer"]:
    """Only an enforced retry decision may resume research."""
    validate_state(state)
    if state["research_status"] == "running":
        if state["current_task"] is None:
            raise ValueError("Critic scheduled research without a current task")
        return "researcher"
    if state["current_task"] is not None:
        raise ValueError("Terminated research still has a current task")
    return "writer"


def research_recursion_limit(settings: Settings) -> int:
    """Bound every legitimate node step, including the final Critic evaluation."""
    return (1 + settings.planner_max_tasks
            + settings.max_retries * settings.critic_max_tasks
            + settings.max_retries + 1 + 1 + 3)


def build_research_graph(
    model: StructuredModel,
    settings: Settings,
    *,
    web_search: Callable[[str], WebSearchResponse],
    retrieve_documents: Callable[[str], list[ResearchMaterial]] | None = None,
    writer_model: StructuredModel | None = None,
    local_document_context: str = "", web_search_available: bool = True,
) -> CompiledStateGraph:
    """Inject dependencies; building the graph makes no model or tool requests."""
    if not web_search_available and retrieve_documents is None:
        raise ValueError("At least one research tool must be available")
    graph = StateGraph(AgentState)
    model = observe_model(model)
    web_search = traced_callable("web_search", "tool", web_search)
    if retrieve_documents is not None:
        retrieve_documents = traced_callable("retrieve_documents", "tool", retrieve_documents)
    graph.add_node("planner", traced_callable("planner", "node", build_planner_node(
        model, settings, local_documents_available=retrieve_documents is not None,
        local_document_context=local_document_context, web_search_available=web_search_available,
    )))
    graph.add_node("researcher", traced_callable("researcher", "node", build_researcher_node(
        model, settings, web_search=web_search, retrieve_documents=retrieve_documents,
    )))
    graph.add_node("critic", traced_callable("critic", "node", build_critic_node(
        model, settings, local_documents_available=retrieve_documents is not None,
        local_document_context=local_document_context, web_search_available=web_search_available,
    )))
    graph.add_node("writer", traced_callable("writer", "node", build_writer_node(
        observe_model(writer_model if writer_model is not None else model), settings)))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_conditional_edges(
        "researcher", route_after_research, {"researcher": "researcher", "critic": "critic"},
    )
    graph.add_conditional_edges(
        "critic", route_after_critic, {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("writer", END)
    return graph.compile()


def run_research(
    user_query: str,
    model: StructuredModel,
    settings: Settings,
    *,
    web_search: Callable[[str], WebSearchResponse],
    retrieve_documents: Callable[[str], list[ResearchMaterial]] | None = None,
    writer_model: StructuredModel | None = None,
    local_document_context: str = "", web_search_available: bool = True,
    research_context: ResearchContext | None = None,
) -> AgentState:
    """Start an independent run; return report, citations and research audit records."""
    graph = build_research_graph(
        model, settings, web_search=web_search, retrieve_documents=retrieve_documents, writer_model=writer_model,
        local_document_context=local_document_context, web_search_available=web_search_available,
    )
    def execute_graph() -> AgentState:
        with span("research.graph", "run") as attributes:
            result = graph.invoke(create_initial_state(user_query, research_context),
                                  config={"recursion_limit": research_recursion_limit(settings)})
            validate_state(result)
            attributes.update(research_status=result["research_status"], report_status=result["report_status"],
                              retry_count=result["retry_count"], evidence_count=len(result["evidence"]))
            attributes["exit_code"] = int(result["research_status"] == "critic_error" or result["report_status"] == "writer_error")
            return result
    if current_recorder() is not None:
        return execute_graph()
    with trace_run(settings.trace_file):
        return execute_graph()
