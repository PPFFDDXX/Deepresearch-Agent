import json
import unittest

from app.agents.researcher import ResearcherError, build_researcher_node
from app.config import Settings
from app.state import ResearchPlan, ResearchTask, create_initial_state, merge_records, validate_state
from app.tools.models import ResearchMaterial, ToolError, WebSearchHit, WebSearchResponse


class Model:
    def __init__(self, outputs=None):
        self.outputs = iter(outputs) if outputs is not None else None
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.outputs is not None:
            return next(self.outputs)
        material = json.loads(kwargs["user"])["materials"][0]
        return json.dumps({"evidence": [{"material_id": material["material_id"],
                                        "claim": "Fixture claim", "excerpt": material["text"]}]})


def apply(state, update):
    result = dict(state)
    for key, value in update.items():
        result[key] = merge_records(state[key], value) if key in (
            "citations", "search_results", "evidence", "research_outcomes",
        ) else value
    validate_state(result)
    return result


class ResearcherTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, researcher_max_attempts=2)
        self.task = ResearchTask(id="t1", question="Fixture question", query="fixture")
        self.state = create_initial_state("Fixture research")
        self.state["research_plan"] = ResearchPlan(tasks=[self.task])
        self.state["current_task"] = self.task

    def search(self, query):
        return WebSearchResponse(query=query, provider="fixture", attempts=1, results=[
            WebSearchHit(title="Fixture", url="https://example.org/page", snippet="Exact source text")])

    def node(self, model=None, search=None, retrieve=None):
        return build_researcher_node(model or Model(), self.settings,
                                     web_search=search or self.search, retrieve_documents=retrieve)

    def test_grounding_and_no_input_mutation(self):
        result = apply(self.state, self.node()(self.state))
        self.assertEqual(self.state["evidence"], [])
        evidence = result["evidence"][0]
        self.assertEqual(evidence.excerpt, "Exact source text")
        self.assertEqual(evidence.origin, "search_snippet")
        self.assertEqual(evidence.citation_id, result["citations"][0].id)
        self.assertIsNone(result["current_task"])

    def test_empty_search_does_not_call_model(self):
        model = Model()
        result = apply(self.state, self.node(model, lambda q: WebSearchResponse(
            query=q, provider="fixture", results=[], attempts=1))(self.state))
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["research_outcomes"][0].status, "no_evidence")

    def test_empty_extraction(self):
        result = apply(self.state, self.node(Model(['{"evidence": []}']))(self.state))
        self.assertEqual(result["evidence"], [])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["research_outcomes"][0].status, "no_evidence")

    def test_fabricated_source_or_excerpt_rejected(self):
        for material_id, excerpt in (("invented", "Exact source text"), ("material-1", "Fabrication")):
            raw = json.dumps({"evidence": [{"material_id": material_id, "claim": "claim", "excerpt": excerpt}]})
            model = Model([raw, raw])
            result = apply(self.state, self.node(model)(self.state))
            self.assertEqual(model.calls, 2)
            self.assertEqual(result["evidence"], [])
            self.assertEqual(result["research_outcomes"][0].status, "extraction_error")
            self.assertEqual(len(result["search_results"]), 1)

    def test_tool_error_recorded(self):
        def fail(query):
            raise ToolError("timeout", "Fixture timeout", attempts=2)
        result = apply(self.state, self.node(search=fail)(self.state))
        self.assertEqual(result["research_outcomes"][0].status, "tool_error")
        self.assertIsNone(result["current_task"])

    def test_local_selection_and_unconfigured_local(self):
        task = self.task.model_copy(update={"tool": "retrieve_documents"})
        self.state.update(research_plan=ResearchPlan(tasks=[task]), current_task=task)
        result = apply(self.state, self.node()(self.state))
        self.assertEqual(result["research_outcomes"][0].status, "tool_error")
        def search_must_not_run(q):
            self.fail("Local task called web search")
        def retrieve(q):
            return [ResearchMaterial(title="Fixture document", locator="fixture.txt", source_type="local",
                                     text="Local excerpt", location="chunk-1", origin="local_chunk")]
        result = apply(self.state, self.node(search=search_must_not_run, retrieve=retrieve)(self.state))
        self.assertEqual(result["evidence"][0].origin, "local_chunk")
        self.assertEqual(result["citations"][0].source_type, "local")

    def test_next_task_and_completed_task_not_repeated(self):
        second = self.task.model_copy(update={"id": "t2"})
        self.state["research_plan"] = ResearchPlan(tasks=[self.task, second])
        model = Model()
        node = self.node(model)
        result = apply(self.state, node(self.state))
        self.assertEqual(result["current_task"], second)
        result["current_task"] = self.task
        result = apply(result, node(result))
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["current_task"], second)
        result = apply(result, node(result))
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(len(result["evidence"]), 2)

    def test_invalid_execution_state(self):
        self.state["current_task"] = None
        with self.assertRaises(ResearcherError):
            self.node()(self.state)


if __name__ == "__main__":
    unittest.main()
