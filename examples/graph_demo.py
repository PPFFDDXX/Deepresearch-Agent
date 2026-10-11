"""Offline multi-task graph fixture; all sources and claims are synthetic."""

import json
from typing import Any

from app.config import Settings
from app.graph import run_research
from examples.researcher_demo import DemoEvidenceModel, demo_search


DEMO_QUERY = "Demonstrate planning and evidence collection across multiple tasks"


class DemoGraphModel(DemoEvidenceModel):
    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        if "executive_summary" in schema["properties"]:
            materials = json.loads(user)["evidence"]
            paragraph = {"text": "Synthetic fixture only: this demonstrates citation traceability, not real research.",
                         "evidence_ids": [item["id"] for item in materials[:2]]}
            return json.dumps({
                "executive_summary": [paragraph], "key_findings": [paragraph],
                "detailed_analysis": [{"title": "Synthetic workflow", "paragraphs": [paragraph]}],
                "conclusion": [paragraph],
            })
        if "sufficient" in schema["properties"]:
            return json.dumps({
                "sufficient": True, "completeness": 1.0, "relevance": 1.0, "source_quality": 1.0,
                "missing_topics": [], "concerns": ["Synthetic fixture: scores are not a real evaluation"],
                "evidence_issues": [], "new_tasks": [], "summary": "Offline fixture only",
            })
        if "tasks" in schema["properties"]:
            return json.dumps({"tasks": [
                {"question": "Inspect synthetic source metadata", "query": "fixture metadata", "tool": "web_search"},
                {"question": "Inspect synthetic evidence traceability", "query": "fixture traceability", "tool": "web_search"},
            ][:schema["properties"]["tasks"]["maxItems"]]})
        return super().generate(system=system, user=user, schema=schema)


def run_demo(settings: Settings) -> dict[str, Any]:
    state = run_research(DEMO_QUERY, DemoGraphModel(), settings, web_search=demo_search)
    return {key: ([item.model_dump() for item in value] if isinstance(value, list)
                  else value.model_dump() if hasattr(value, "model_dump") else value)
            for key, value in state.items()}
