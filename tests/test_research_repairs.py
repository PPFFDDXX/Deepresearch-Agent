"""Offline regressions for coverage diagnostics, retrieval and planning scope."""
import json
import unittest

from app.agents.critic import evaluate_evidence, CriticError
from app.agents.planner import plan_research
from app.config import PROJECT_ROOT, Settings
from app.coverage import CoverageValidationError, validate_requirement_coverage
from app.plan_scope import review_task_scope
from app.rag.pipeline import build_local_rag
from app.research_context import ResearchContext
from app.state import ResearchTask, ResearchPlan, RequirementCoverage
from app.tools.models import ResearchMaterial, ToolError
from app.tools.requirement_retrieval import build_requirement_queries, retrieve_for_requirements
from test_requirement_coverage import BudgetModel
import test_requirement_coverage as coverage_fixtures


class ResearchRepairTests(unittest.TestCase):
    def settings(self):
        return Settings(_env_file=None, embedding_provider="hashing")

    def state(self):
        return coverage_fixtures.RequirementTests().run_budget(BudgetModel())

    def test_unknown_and_rejected_ids_have_distinct_safe_diagnostics(self):
        state = self.state()
        row = state["critic_assessments"][-1].requirement_coverage[0]
        with self.assertRaises(CoverageValidationError) as unknown:
            validate_requirement_coverage(state, [row.model_copy(update={"evidence_ids": ["SECRET-unknown"]})], set(), require_complete=False)
        self.assertEqual(unknown.exception.code, "unknown_evidence")
        self.assertNotIn("SECRET", str(unknown.exception))
        with self.assertRaises(CoverageValidationError) as rejected:
            validate_requirement_coverage(state, [row], set(), require_complete=False)
        self.assertEqual(rejected.exception.code, "rejected_evidence")
        self.assertEqual(rejected.exception.task_id, "task-1")
        self.assertEqual(rejected.exception.requirement_id, "req-1")
        self.assertIn("covered=false", str(rejected.exception))

    def test_failed_critic_preserves_each_attempt_diagnostic(self):
        class ConflictingModel(BudgetModel):
            def generate(self, **kwargs):
                raw = super().generate(**kwargs)
                if "sufficient" in kwargs["schema"]["properties"]:
                    draft = json.loads(raw)
                    draft["evidence_issues"] = [{"evidence_id": draft["requirement_coverage"][0]["evidence_ids"][0],
                        "category": "irrelevant", "reason": "Fixture rejected support"}]
                    return json.dumps(draft)
                return raw
        state = coverage_fixtures.RequirementTests().run_budget(ConflictingModel())
        assessment = state["critic_assessments"][-1]
        self.assertEqual(state["research_status"], "critic_error")
        self.assertEqual([item.attempt for item in assessment.diagnostics], [1, 2])
        self.assertEqual([item.code for item in assessment.diagnostics], ["rejected_evidence"] * 2)
        self.assertTrue(all(item.evidence_ids for item in assessment.diagnostics))

    def test_duplicate_support_normalized_without_weakening_rejection(self):
        class DuplicateModel(BudgetModel):
            def generate(self, **kwargs):
                raw = super().generate(**kwargs)
                if "sufficient" in kwargs["schema"]["properties"]:
                    draft = json.loads(raw)
                    for row in draft["requirement_coverage"]:
                        row["evidence_ids"] *= 2
                    return json.dumps(draft)
                return raw
        state = coverage_fixtures.RequirementTests().run_budget(DuplicateModel())
        assessment = state["critic_assessments"][-1]
        self.assertEqual(state["report_status"], "generated")
        self.assertEqual(len(assessment.diagnostics), 2)
        self.assertEqual(assessment.diagnostics[0].code, "duplicate_evidence_normalized")
        self.assertTrue(all(len(row.evidence_ids) == 1 for row in assessment.requirement_coverage))

    def test_unknown_reference_repair_feedback_and_audit(self):
        state = self.state()
        class UnknownThenValid(BudgetModel):
            def __init__(self):
                super().__init__()
                self.critic_calls = 0
            def generate(self, **kwargs):
                draft = json.loads(super().generate(**kwargs))
                if "sufficient" in kwargs["schema"]["properties"]:
                    self.critic_calls += 1
                    if self.critic_calls == 1:
                        draft["requirement_coverage"][0]["evidence_ids"] = ["invented"]
                    else:
                        payload = json.loads(kwargs["user"])
                        self.feedback = payload["validation_feedback"]
                return json.dumps(draft)
        model = UnknownThenValid()
        audit = []
        draft = evaluate_evidence(state, model, self.settings(), diagnostics=audit)
        self.assertEqual(len(draft.requirement_coverage), 2)
        self.assertEqual(audit[0].code, "unknown_evidence")
        self.assertIn("task=task-1 requirement=req-1", model.feedback)
        self.assertNotIn("invented", model.feedback)

    def budget_task(self):
        return ResearchTask(id="task-1", question="知舟试点的预算详情是什么？", query="知舟试点预算说明",
            tool="retrieve_documents", requirements=[{"id": f"req-{number}", "description": text}
                for number, text in enumerate(["知舟试点预算总额", "预算分配明细", "预算使用期限", "预算审批流程"], 1)])

    def test_real_documents_focused_queries_recover_budget_sections(self):
        rag = build_local_rag([PROJECT_ROOT / "examples/doc"], self.settings())
        task = self.budget_task()
        hits, attempts = retrieve_for_requirements(task, rag.retrieve_documents, limit=8)
        text = " ".join(hit.text for hit in hits if hit.locator.endswith("04-pilot-budget.txt"))
        for term in ("18,000", "6,000", "5,000", "3,000", "审批", "实际支出"):
            self.assertIn(term, text)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(build_requirement_queries(task)[1], "预算分配明细")
        self.assertEqual(len(hits), len({(hit.locator, hit.location, hit.text) for hit in hits}))

    def test_retrieval_fair_merge_and_partial_tool_failure(self):
        task = self.budget_task()
        def item(text):
            return ResearchMaterial(title="fixture", locator="/fixture.md", text=text,
                source_type="local", origin="local_chunk", location=text)
        def retrieve(query):
            if query == "预算使用期限":
                raise ToolError("timeout", "SECRET")
            return [item(query), item("shared")]
        hits, attempts = retrieve_for_requirements(task, retrieve, limit=3)
        self.assertEqual([hit.text for hit in hits], ["知舟试点预算总额", "预算分配明细", "预算审批流程"])
        self.assertEqual(attempts[2].error_code, "timeout")
        self.assertNotIn("SECRET", str(attempts))

    def test_scope_review_filters_extras_and_preserves_explicit_unknown_request(self):
        tasks = [
            {"question": "预算", "query": "预算", "tool": "retrieve_documents", "requirements": ["预算总额", "预算调整机制", "实际支出"]},
            {"question": "验收", "query": "验收", "tool": "retrieve_documents", "requirements": ["验收结果"]}]
        class ScopeModel:
            def generate(self, **kwargs):
                self.prompt = kwargs["system"]
                return json.dumps({"task_reviews": [
                    {"index": 0, "keep": True, "reason": "Requested budget and actual spend", "requirements": [
                        {"index": 0, "keep": True, "reason": "Requested"},
                        {"index": 1, "keep": False, "reason": "Unrequested generic mechanism"},
                        {"index": 2, "keep": True, "reason": "Explicit question, even without records"}]},
                    {"index": 1, "keep": False, "reason": "Unrequested acceptance topic", "requirements": [
                        {"index": 0, "keep": False, "reason": "Unrequested"}]}]})
        model = ScopeModel()
        filtered, audit = review_task_scope("预算及实际支出是什么？", tasks, model, ResearchContext())
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["requirements"], ["预算总额", "实际支出"])
        self.assertFalse(audit[1].kept)
        self.assertIn("lack of documents/evidence", model.prompt)

    def test_scope_review_must_account_for_every_index(self):
        class MissingScopeModel:
            def generate(self, **kwargs):
                return '{"task_reviews": []}'
        with self.assertRaisesRegex(ValueError, "every task index"):
            review_task_scope("预算", [{"question": "预算", "requirements": ["金额"]}], MissingScopeModel(), ResearchContext())


if __name__ == "__main__":
    unittest.main()
