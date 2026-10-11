"""Application fallback for explicit uncovered requirements; no LLM or new topics."""
from app.state import AgentState, ResearchTask, RequirementCoverage


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def automatic_refinements(state: AgentState, coverage: list[RequirementCoverage], *,
                          limit: int, local_available: bool, web_available: bool) -> list[ResearchTask]:
    """Group identical gaps and choose a finite, previously unexecuted query variant."""
    tasks = {task.id: task for task in state["research_plan"].tasks}
    seen = {(task.tool, normalize_query(task.query)) for task in tasks.values()}
    seen.update((tasks[outcome.task_id].tool, normalize_query(attempt.query))
                for outcome in state["research_outcomes"] for attempt in outcome.retrieval_attempts)
    groups: dict[tuple[str, str], tuple[str, list[str]]] = {}
    for row in coverage:
        if row.covered:
            continue
        parent = tasks[row.task_id]
        tool = parent.tool
        if ((tool == "retrieve_documents" and not local_available)
                or (tool == "web_search" and not web_available)):
            continue
        requirement = next(item for item in parent.requirements if item.id == row.requirement_id)
        key = (tool, normalize_query(requirement.description))
        if key not in groups:
            groups[key] = (requirement.description, [])
        if parent.id not in groups[key][1]:
            groups[key][1].append(parent.id)
    result = []
    for (tool, _), (description, parents) in groups.items():
        if len(result) >= limit:
            break
        variants = [description, f"{description} 详细规定 原文", f"{description} 具体内容 章节"]
        query = next((value for value in variants if (tool, normalize_query(value)) not in seen), None)
        if query is None:
            continue
        seen.add((tool, normalize_query(query)))
        result.append(ResearchTask(id=f"retry-{state['retry_count'] + 1}-task-{len(result) + 1}",
            question=description, query=query, tool=tool, addresses_task_ids=tuple(parents),
            requirements=[{"id": "req-1", "description": description}], refinement_origin="automatic"))
    return result
