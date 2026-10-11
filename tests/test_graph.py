"""Integration tests through real LangGraph with offline model/tool dependencies."""

import json
import unittest

from app.agents.planner import PlannerError
from app.config import Settings
from app.graph import build_research_graph, run_research
from app.state import create_initial_state, validate_state
from app.tools.models import ToolError, WebSearchResponse
from examples.graph_demo import DEMO_QUERY, DemoGraphModel
from examples.researcher_demo import demo_search


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None)

    def run_graph(self, search=demo_search, model=None, settings=None):
        return run_research(DEMO_QUERY, model or DemoGraphModel(), settings or self.settings, web_search=search)

    def test_multiple_tasks_merge_sources_and_preserve_input(self):
        calls = []
        def search(query):
            calls.append(query)
            return demo_search(query)
        initial = create_initial_state(DEMO_QUERY)
        graph = build_research_graph(DemoGraphModel(), self.settings, web_search=search)
        result = graph.invoke(initial)
        validate_state(result)
        self.assertIsNone(initial["research_plan"])
        self.assertEqual(initial["evidence"], [])
        self.assertEqual(calls, [task.query for task in result["research_plan"].tasks])
        self.assertEqual(len(result["research_outcomes"]), 2)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(len(result["search_results"]), 2)
        self.assertEqual(len(result["citations"]), 1)
        self.assertIsNone(result["current_task"])
        self.assertIn("## Sources / Citations", result["final_report"])
        self.assertEqual(result["report_status"], "generated")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["research_status"], "sufficient")
        self.assertEqual(result["critic_assessments"][-1].decision, "sufficient")

    def test_empty_and_failed_tools_advance_to_next_task(self):
        calls = []
        def search(query):
            calls.append(query)
            if len(calls) == 1:
                raise ToolError("timeout", "Fixture timeout")
            return WebSearchResponse(query=query, provider="fixture", attempts=1, results=[])
        state = self.run_graph(search)
        self.assertEqual([o.status for o in state["research_outcomes"]], ["tool_error", "no_evidence"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(state["evidence"], [])
        self.assertIsNone(state["current_task"])
        self.assertEqual(state["research_status"], "insufficient")

    def test_extraction_failure_does_not_skip_remaining_tasks(self):
        class InvalidEvidenceModel(DemoGraphModel):
            def generate(self, **kwargs):
                if any(key in kwargs["schema"]["properties"] for key in ("tasks", "sufficient", "executive_summary")):
                    return super().generate(**kwargs)
                return "invalid JSON"
        state = self.run_graph(model=InvalidEvidenceModel())
        self.assertEqual([o.status for o in state["research_outcomes"]], ["extraction_error"] * 2)
        self.assertEqual(len(state["search_results"]), 2)
        self.assertEqual(state["evidence"], [])
        self.assertEqual(state["research_status"], "insufficient")

    def test_failed_planning_never_calls_search(self):
        class InvalidPlanModel:
            def generate(self, **kwargs):
                return json.dumps({"tasks": []})
        def search(query):
            self.fail("Search must not run after planning failure")
        with self.assertRaises(PlannerError):
            self.run_graph(search, InvalidPlanModel())

    def test_independent_invocations_do_not_share_records(self):
        graph = build_research_graph(DemoGraphModel(), self.settings, web_search=demo_search)
        first = graph.invoke(create_initial_state(DEMO_QUERY))
        second = graph.invoke(create_initial_state("Another independent query"))
        self.assertEqual(len(first["evidence"]), 2)
        self.assertEqual(len(second["evidence"]), 2)
        self.assertEqual(second["user_query"], "Another independent query")

    def test_single_and_maximum_task_counts_terminate(self):
        class ManyTasksModel(DemoGraphModel):
            def generate(self, **kwargs):
                if "tasks" in kwargs["schema"]["properties"]:
                    count = kwargs["schema"]["properties"]["tasks"]["maxItems"]
                    return json.dumps({"tasks": [
                        {"question": f"Question {i}", "query": f"Query {i}", "tool": "web_search"}
                        for i in range(count)
                    ]})
                return super().generate(**kwargs)
        for count in (1, 10):
            with self.subTest(count=count):
                state = self.run_graph(model=ManyTasksModel(), settings=Settings(_env_file=None, planner_max_tasks=count))
                self.assertEqual(len(state["research_outcomes"]), count)
                self.assertIsNone(state["current_task"])


if __name__ == "__main__":
    unittest.main()
