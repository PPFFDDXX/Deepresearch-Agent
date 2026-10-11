"""Offline fixture demonstrating insufficient -> refinement -> sufficient."""

import json
from typing import Any

from app.config import Settings
from app.graph import run_research
from examples.graph_demo import DEMO_QUERY, DemoGraphModel
from examples.researcher_demo import demo_search


class DemoCriticModel(DemoGraphModel):
    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        if "sufficient" not in schema["properties"]:
            return super().generate(system=system, user=user, schema=schema)
        first_round = json.loads(user)["round"] == 0
        return json.dumps({
            "sufficient": not first_round,
            "completeness": 0.5 if first_round else 1.0, "relevance": 1.0, "source_quality": 1.0,
            "missing_topics": ["Synthetic follow-up coverage"] if first_round else [],
            "concerns": ["Synthetic fixture only; these scores do not evaluate real evidence"],
            "evidence_issues": [],
            "new_tasks": [{"question": "Inspect synthetic follow-up material", "query": "fixture follow-up",
                           "tool": "web_search", "addresses_task_ids": ["task-1"]}] if first_round else [],
            "summary": "Offline refinement fixture",
        })


def run_demo(settings: Settings) -> dict[str, Any]:
    state = run_research(DEMO_QUERY, DemoCriticModel(), settings, web_search=demo_search)
    return {key: ([item.model_dump() for item in value] if isinstance(value, list)
                  else value.model_dump() if hasattr(value, "model_dump") else value)
            for key, value in state.items()}
