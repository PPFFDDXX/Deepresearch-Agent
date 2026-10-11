"""Bounded local multi-query retrieval with fair chunk fusion and per-query audit."""
from collections.abc import Callable

from app.state import ResearchTask, RetrievalAttempt
from app.tools.models import ResearchMaterial, ToolError


def build_requirement_queries(task: ResearchTask) -> list[str]:
    """At most ten focused requirement queries plus one original query."""
    if task.refinement_origin == "automatic":
        # The fallback already selected a novel requirement-focused query; do not
        # repeat the original requirement query before it on every research round.
        return [task.query]
    # Repeating a document title can drown out section keywords and return preambles.
    queries = [item.description for item in task.requirements] + [task.query]
    seen = set()
    result = []
    for query in queries:
        key = " ".join(query.casefold().split())
        if key not in seen:
            seen.add(key)
            result.append(query)
    return result


def retrieve_for_requirements(task: ResearchTask, retrieve: Callable[[str], list[ResearchMaterial]],
                              *, limit: int) -> tuple[list[ResearchMaterial], list[RetrievalAttempt]]:
    """Fuse by result rank across queries, not by concatenating the first query's hits."""
    if limit < 1:
        raise ValueError("Material limit must be positive")
    batches = []
    attempts = []
    for query in build_requirement_queries(task):
        try:
            hits = retrieve(query)
            if not isinstance(hits, list) or any(not isinstance(item, ResearchMaterial)
                or item.source_type != "local" or item.origin != "local_chunk" for item in hits):
                raise ToolError("invalid_response", "Local retriever must return local chunks")
            batches.append(hits[:limit])
            attempts.append(RetrievalAttempt(query=query, result_count=len(hits), status="success"))
        except ToolError as exc:
            batches.append([])
            attempts.append(RetrievalAttempt(query=query, result_count=0, status="tool_error", error_code=exc.code))
    merged = []
    seen = set()
    for rank in range(max((len(batch) for batch in batches), default=0)):
        for batch in batches:
            if rank < len(batch):
                item = batch[rank]
                key = (item.locator, item.location, item.text)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
                    if len(merged) >= limit:
                        return merged, attempts
    return merged, attempts
