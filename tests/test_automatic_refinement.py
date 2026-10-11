"""Automatic refinement must recover or terminate within the existing budgets."""
import json
import unittest

from app.agents.critic import build_critic_node
from app.agents.refinement import automatic_refinements
from app.config import Settings
from app.graph import run_research
from app.research_context import ResearchContext
from app.state import ResearchPlan, ResearchTask, RequirementCoverage, create_initial_state, validate_state
from app.tools.models import ResearchMaterial, WebSearchResponse
from app.tools.requirement_retrieval import build_requirement_queries
from test_requirement_coverage import BudgetModel


class MissingThenRecovered(BudgetModel):
    def __init__(self, recover=True):
        super().__init__()
        self.recover = recover

    def generate(self, **kwargs):
        if "sufficient" not in kwargs["schema"]["properties"]:
            return super().generate(**kwargs)
        payload = json.loads(kwargs["user"])
        covered = self.recover and payload["round"] > 0
        evidence = payload["evidence"][0]["id"]
        return json.dumps({"sufficient": True, "completeness": 1, "relevance": 1, "source_quality": 1,
            "missing_topics": [], "concerns": [], "evidence_issues": [], "new_tasks": [], "summary": "Fixture",
            "requirement_coverage": [{"task_id": task["id"], "requirement_id": req["id"],
                "covered": covered or req["description"] == "预算是否代表实际支出",
                "evidence_ids": [evidence] if covered or req["description"] == "预算是否代表实际支出" else [],
                "reason": "Supported" if covered else "Missing requirement"}
                for task in payload["research_plan"]["tasks"] for req in task["requirements"]]})


class AutomaticRefinementTests(unittest.TestCase):
    def run_graph(self, model, retries=2):
        calls = []
        def retrieve(query):
            calls.append(query)
            return [ResearchMaterial(title="预算", locator="/fixture.md", text="预算上限为人民币18,000元。预算不是实际支出。",
                source_type="local", origin="local_chunk", location="第1节")]
        state = run_research("预算与实际支出是什么？", model,
            Settings(_env_file=None, max_retries=retries, embedding_provider="hashing"),
            web_search=lambda q: WebSearchResponse(query=q, provider="fixture", attempts=1, results=[]),
            retrieve_documents=retrieve, web_search_available=False,
            research_context=ResearchContext(scope="document_grounded"))
        return state, calls

    def test_empty_model_tasks_trigger_research_and_recover(self):
        state, calls = self.run_graph(MissingThenRecovered())
        validate_state(state)
        self.assertEqual(state["research_status"], "sufficient")
        self.assertEqual(state["retry_count"], 1)
        fallback = state["research_plan"].tasks[-1]
        self.assertEqual(fallback.refinement_origin, "automatic")
        self.assertEqual(fallback.addresses_task_ids, ("task-1",))
        self.assertEqual(build_requirement_queries(fallback), [fallback.query])
        self.assertEqual(calls.count(fallback.query), 1)
        self.assertEqual(state["critic_assessments"][0].decision, "retry")

    def test_persistent_gap_terminates_at_retry_limit(self):
        state, calls = self.run_graph(MissingThenRecovered(recover=False))
        validate_state(state)
        self.assertEqual(state["retry_count"], 2)
        self.assertEqual(state["research_status"], "insufficient")
        self.assertEqual(state["critic_assessments"][-1].decision, "exhausted")
        self.assertEqual(len(state["research_plan"].tasks), 3)
        automatic = state["research_plan"].tasks[1:]
        self.assertEqual(len({task.query for task in automatic}), 2)
        self.assertTrue(all(task.requirements[0].description == "预算总額及币种" for task in automatic))

    def test_no_remaining_budget_does_not_schedule(self):
        state, calls = self.run_graph(MissingThenRecovered(recover=False), retries=0)
        self.assertEqual(len(state["research_plan"].tasks), 1)
        self.assertEqual(state["critic_assessments"][-1].decision, "exhausted")

    def state_and_rows(self):
        state = create_initial_state("预算与规范")
        state["research_plan"] = ResearchPlan(tasks=[ResearchTask(id="task-1", question="预算", query="预算",
            tool="retrieve_documents", requirements=[{"id": f"req-{i}", "description": f"预算要求{i}"} for i in range(4)])])
        rows = [RequirementCoverage(task_id="task-1", requirement_id=f"req-{i}", covered=False,
            evidence_ids=[], reason="Missing") for i in range(4)]
        return state, rows

    def test_capacity_tools_and_finite_queries(self):
        state, rows = self.state_and_rows()
        self.assertEqual(automatic_refinements(state, rows, limit=3, local_available=False, web_available=True), [])
        tasks = automatic_refinements(state, rows, limit=3, local_available=True, web_available=False)
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(task.tool == "retrieve_documents" for task in tasks))
        for number, query in enumerate(("预算要求0", "预算要求0 详细规定 原文", "预算要求0 具体内容 章节"), 1):
            state["research_plan"] = ResearchPlan(tasks=[*state["research_plan"].tasks,
                ResearchTask(id=f"used-{number}", question="已有查询", query=query, tool="retrieve_documents")])
        self.assertEqual(automatic_refinements(state, rows[:1], limit=3, local_available=True, web_available=False), [])

    def test_duplicate_gap_descriptions_merge_parent_links(self):
        state, rows = self.state_and_rows()
        parent = state["research_plan"].tasks[0]
        state["research_plan"] = ResearchPlan(tasks=[parent, ResearchTask(id="retry-1-task-1", question="预算",
            query="old followup", tool="retrieve_documents", addresses_task_ids=(parent.id,), requirements=[parent.requirements[0]])])
        rows = [rows[0], rows[0].model_copy(update={"task_id": "retry-1-task-1"})]
        tasks = automatic_refinements(state, rows, limit=3, local_available=True, web_available=False)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].addresses_task_ids, ("task-1", "retry-1-task-1"))


if __name__ == "__main__":
    unittest.main()
