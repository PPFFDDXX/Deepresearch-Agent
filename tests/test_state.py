"""Offline data-contract tests, including actual LangGraph reducer behavior."""

import unittest

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.state import (
    AgentState, Citation, Evidence, ResearchPlan, ResearchTask, StateUpdate,
    create_initial_state, merge_records, validate_state,
)


class StateTests(unittest.TestCase):
    def test_initial_state_is_independent(self):
        first = create_initial_state(" question ")
        second = create_initial_state("other")
        self.assertEqual(first["user_query"], "question")
        self.assertIsNot(first["evidence"], second["evidence"])
        validate_state(first)
        with self.assertRaises(ValueError):
            create_initial_state(" ")

    def test_record_validation(self):
        with self.assertRaises(ValidationError):
            ResearchTask(id="t1", question=" ", query="query")
        task = ResearchTask(id="t1", question="question", query="query")
        with self.assertRaises(ValidationError):
            ResearchPlan(tasks=[task, task])

    def test_reducer_replay_and_conflict(self):
        source = Citation(id="s1", title="Source", source_type="web", locator="https://example.org")
        previous = [source]
        self.assertEqual(merge_records(previous, [source]), [source])
        self.assertEqual(previous, [source])
        with self.assertRaises(ValueError):
            merge_records(previous, [source.model_copy(update={"title": "Changed"})])

    def test_graph_updates_and_source_integrity(self):
        task = ResearchTask(id="t1", question="question", query="query")
        source = Citation(id="s1", title="Fixture source", source_type="local", locator="fixture.txt")
        evidence = Evidence(id="e1", task_id="t1", citation_id="s1", claim="Fixture claim", excerpt="Fixture excerpt")

        def first(state: AgentState) -> StateUpdate:
            return {"research_plan": ResearchPlan(tasks=[task]), "current_task": task,
                    "citations": [source], "evidence": [evidence]}

        def second(state: AgentState) -> StateUpdate:
            return {"citations": [source], "retry_count": 1, "current_task": None}

        graph = StateGraph(AgentState)
        graph.add_node("first", first)
        graph.add_node("second", second)
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        result = graph.compile().invoke(create_initial_state("question"))
        validate_state(result)
        self.assertEqual(result["citations"], [source])
        self.assertEqual(result["evidence"], [evidence])
        self.assertEqual(result["retry_count"], 1)
        self.assertIsNone(result["current_task"])
        with self.assertRaises(ValueError):
            validate_state({**result, "citations": []})
        with self.assertRaises(ValueError):
            validate_state({**result, "retry_count": -1})


if __name__ == "__main__":
    unittest.main()
