"""Synthetic one-task demo; no model or tool network calls."""

import json
from typing import Any

from app.agents.researcher import build_researcher_node
from app.config import Settings
from app.state import ResearchPlan, ResearchTask, create_initial_state, merge_records, validate_state
from app.tools.models import WebSearchHit, WebSearchResponse


def demo_search(query: str) -> WebSearchResponse:
    return WebSearchResponse(query=query, provider="offline-fixture", attempts=1, results=[
        WebSearchHit(title="Synthetic fixture — not a real source", url="https://example.org/fixture",
                     snippet="Synthetic snippet demonstrating evidence traceability. No research was performed."),
    ])


class DemoEvidenceModel:
    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        material = json.loads(user)["materials"][0]
        return json.dumps({"evidence": [{
            "material_id": material["material_id"],
            "claim": "This synthetic fixture demonstrates evidence traceability only.",
            "excerpt": material["text"],
        }]})


def run_demo(settings: Settings) -> dict[str, Any]:
    task = ResearchTask(id="demo-task", question="Inspect the synthetic search fixture", query="fixture")
    state = create_initial_state(task.question)
    state["research_plan"] = ResearchPlan(tasks=[task])
    state["current_task"] = task
    node = build_researcher_node(DemoEvidenceModel(), settings, web_search=demo_search)
    update = node(state)
    # No real graph until Phase 6: explicitly demonstrate the same merge rules.
    for key, value in update.items():
        state[key] = merge_records(state[key], value) if key in (
            "citations", "search_results", "evidence", "research_outcomes",
        ) else value
    validate_state(state)
    return {"evidence": [e.model_dump() for e in state["evidence"]],
            "citations": [c.model_dump() for c in state["citations"]],
            "research_outcomes": [o.model_dump() for o in state["research_outcomes"]],
            "current_task": state["current_task"]}
