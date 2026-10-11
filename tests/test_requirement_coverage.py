"""Offline regression cases for demo scope, semantic coverage and Writer repair."""
import json
import unittest
from pathlib import Path

from app.agents.planner import PlannerError, plan_research
from app.config import PROJECT_ROOT, Settings
from app.coverage import validate_requirement_coverage, validate_report_coverage
from app.coverage import uncovered_requirement_tasks
from app.graph import run_research
from app.rag.pipeline import build_local_rag
from app.research_context import ResearchContext, build_research_context
from app.state import RequirementCoverage, ResearchTask, ResearchPlan, validate_state
from app.tools.models import ResearchMaterial, WebSearchResponse


class BudgetModel:
    def __init__(self, *, omit_first=False, always_omit=False, missing_amount=False, scope_bad=False):
        self.omit_first = omit_first
        self.always_omit = always_omit
        self.missing_amount = missing_amount
        self.scope_bad = scope_bad
        self.writer_calls = 0
        self.review_calls = 0
        self.payloads = []

    def generate(self, *, system, user, schema):
        payload = json.loads(user)
        self.payloads.append((system, payload))
        properties = schema["properties"]
        if "task_reviews" in properties:
            return json.dumps({"task_reviews": [{"index": task["index"], "keep": True, "reason": "Requested budget",
                "requirements": [{"index": index, "keep": True, "reason": "Requested obligation"}
                                 for index in range(len(task["requirements"]))]} for task in payload["tasks"]]})
        if "tasks" in properties:
            return json.dumps({"tasks": [{"question": "预算是什么？", "query": "预算金额 实际支出",
                "tool": "retrieve_documents", "requirements": ["预算总額及币种", "预算是否代表实际支出"]}]})
        if "sufficient" in properties:
            evidence = payload["evidence"]
            return json.dumps({"sufficient": True, "completeness": 1, "relevance": 1,
                "source_quality": 1, "missing_topics": [], "concerns": [], "evidence_issues": [],
                "new_tasks": [], "summary": "Document-grounded synthetic facts are supported",
                "requirement_coverage": [{"task_id": "task-1", "requirement_id": f"req-{number}",
                    "covered": not (self.missing_amount and number == 1),
                    "evidence_ids": [] if self.missing_amount and number == 1 else [evidence[0]["id"]],
                    "reason": "No amount support" if self.missing_amount and number == 1 else "Exact document support"}
                    for number in (1, 2)]})
        if "executive_summary" in properties:
            self.writer_calls += 1
            omit = self.always_omit or self.omit_first and self.writer_calls == 1
            text = "依据虚构演示文档，预算不是实际支出。" if omit else "依据虚构演示文档，预算上限为人民币18,000元，预算不是实际支出。"
            paragraph = {"text": text, "evidence_ids": [payload["evidence"][0]["id"]]}
            return json.dumps({"executive_summary": [paragraph], "key_findings": [paragraph],
                "detailed_analysis": [{"title": "预算", "paragraphs": [paragraph]}], "conclusion": [paragraph]})
        if "scope_valid" in properties:
            self.review_calls += 1
            has_amount = "18,000" in payload["paragraphs"][0]["text"]
            return json.dumps({"scope_valid": not self.scope_bad, "scope_reason": "Scope expanded" if self.scope_bad else "Attributed demo facts",
                "coverage": [{"task_id": row["task_id"], "requirement_id": row["requirement_id"],
                    "covered": has_amount or row["requirement_id"] == "req-2",
                    "paragraph_indices": [0] if has_amount or row["requirement_id"] == "req-2" else [],
                    "reason": "Explicit body answer" if has_amount else "Budget amount omitted"}
                    for row in payload["requirements"]]})
        material = payload["materials"][0]
        return json.dumps({"evidence": [{"material_id": material["material_id"], "claim": "演示预算金额与实际支出边界",
            "excerpt": material["text"]}]})


class RequirementTests(unittest.TestCase):
    def settings(self, **changes):
        return Settings(_env_file=None, embedding_provider="hashing", max_retries=0, **changes)

    def run_budget(self, model):
        locator = str(PROJECT_ROOT / "examples/doc/04-pilot-budget.txt")
        context = ResearchContext(scope="document_grounded", demo_source_locators=(locator,))
        return run_research("预算与实际支出是什么？", model, self.settings(),
            web_search=lambda q: WebSearchResponse(query=q, provider="fixture", attempts=1, results=[]),
            retrieve_documents=lambda q: [ResearchMaterial(title="演示预算", locator=locator, source_type="local",
                origin="local_chunk", text="虚构演示资料。预算上限为人民币18,000元。预算不是实际支出。", location="第1节")],
            web_search_available=False, research_context=context)

    def test_demo_budget_is_attributed_not_discarded(self):
        model = BudgetModel()
        state = self.run_budget(model)
        validate_state(state)
        self.assertEqual(state["research_status"], "sufficient")
        self.assertEqual(state["report_status"], "generated")
        self.assertIn("18,000", state["final_report"])
        self.assertIn("不证明真实企业预算", state["final_report"])
        self.assertEqual(len(state["report_coverage"]), 2)
        critic_prompt = next(system for system, payload in model.payloads if "research_plan" in payload)
        self.assertIn("Do NOT reject a claim solely", critic_prompt)

    def test_writer_body_omission_triggers_bounded_repair(self):
        model = BudgetModel(omit_first=True)
        state = self.run_budget(model)
        self.assertEqual(model.writer_calls, 2)
        self.assertEqual(model.review_calls, 2)
        self.assertEqual(state["report_status"], "generated")
        writer_payloads = [payload for system, payload in model.payloads if "critic_concerns" in payload]
        self.assertIn("omitted", writer_payloads[-1]["validation_feedback"])

    def test_unrepaired_body_is_not_published(self):
        model = BudgetModel(always_omit=True)
        state = self.run_budget(model)
        self.assertEqual(model.writer_calls, 2)
        self.assertEqual(state["report_status"], "writer_error")
        self.assertIsNone(state["report_draft"])
        self.assertEqual(state["report_citations"], [])

    def test_scope_expansion_fails_writer_review(self):
        state = self.run_budget(BudgetModel(scope_bad=True))
        self.assertEqual(state["report_status"], "writer_error")
        self.assertIn("source scope", state["report_error"])

    def test_task_evidence_association_is_not_coverage(self):
        state = self.run_budget(BudgetModel(missing_amount=True))
        self.assertEqual(state["research_status"], "insufficient")
        self.assertEqual(state["critic_assessments"][-1].uncovered_task_ids, ["task-1"])
        self.assertIn("Uncovered requirement", state["final_report"])
        self.assertEqual(state["report_status"], "partial")

    def test_refinement_link_does_not_automatically_cover_budget_parent(self):
        state = self.run_budget(BudgetModel())
        plan = state["research_plan"]
        refinement = ResearchTask(id="retry-1-task-1", question="试点用户范围", query="试点用户",
            tool="retrieve_documents", addresses_task_ids=("task-1",),
            requirements=[{"id": "req-1", "description": "试点用户范围"}])
        state = {**state, "research_plan": ResearchPlan(tasks=[*plan.tasks, refinement])}
        rows = [row.model_copy(update={"covered": False, "evidence_ids": []})
                for row in state["critic_assessments"][-1].requirement_coverage]
        rows.append(RequirementCoverage(task_id=refinement.id, requirement_id="req-1", covered=True,
            evidence_ids=state["critic_assessments"][-1].accepted_evidence_ids, reason="Only user scope supported"))
        self.assertEqual(uncovered_requirement_tasks(state, rows), ["task-1"])

    def test_actual_rag_budget_supports_scoped_report_end_to_end(self):
        rag = build_local_rag([PROJECT_ROOT / "examples/doc"], self.settings())
        # Use a focused golden query; general-query retrieval tuning is a separate concern.
        state = run_research("预算与实际支出是什么？", BudgetModel(), self.settings(),
            web_search=lambda q: WebSearchResponse(query=q, provider="fixture", attempts=1, results=[]),
            retrieve_documents=lambda q: rag.retrieve_documents("三个月试点预算上限 人民币"),
            local_document_context=rag.describe_documents(), web_search_available=False,
            research_context=build_research_context(rag.documents, local_only=True))
        validate_state(state)
        self.assertEqual(state["report_status"], "generated")
        self.assertIn("18,000", state["final_report"])

    def test_unknown_rejected_duplicate_and_empty_support_fail(self):
        state = self.run_budget(BudgetModel())
        row = state["critic_assessments"][-1].requirement_coverage[0]
        accepted = set(state["critic_assessments"][-1].accepted_evidence_ids)
        for rows in ([row], [row, row], [row.model_copy(update={"evidence_ids": ["invented"]})],
                     [row.model_copy(update={"evidence_ids": []})]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_requirement_coverage(state, rows, accepted)

    def test_report_indices_and_citation_membership_are_checked(self):
        state = self.run_budget(BudgetModel())
        row = state["report_coverage"][0]
        for indices in ([99], [-1], [0, 0], []):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                validate_report_coverage(state, state["report_draft"], [row.model_copy(update={"paragraph_indices": indices})])

    def test_context_recognizes_repository_examples_not_text_markers(self):
        rag = build_local_rag([PROJECT_ROOT / "examples/doc"], self.settings())
        context = build_research_context(rag.documents, local_only=True)
        self.assertEqual(len(context.demo_source_locators), 4)
        uploaded = rag.documents[0].model_copy(update={"locator": str(PROJECT_ROOT / "reports/uploaded-demo.md")})
        self.assertEqual(build_research_context([uploaded], local_only=True).demo_source_locators, ())

    def test_strict_planner_cannot_silently_omit_requirements(self):
        class InvalidPlanner:
            def generate(self, **kwargs):
                return json.dumps({"tasks": [{"question": "预算", "query": "预算", "tool": "web_search"}]})
        with self.assertRaises(PlannerError):
            plan_research("预算", InvalidPlanner(), self.settings(), research_context=ResearchContext())


if __name__ == "__main__":
    unittest.main()
