"""Regression: oversized summaries must be repairable without dropping coverage."""
import json
import unittest

from pydantic import ValidationError

from app.agents.critic import CriticError, EvaluationDraft, evaluate_evidence, evaluation_validation_feedback
from app.agents.planner import plan_research
from app.config import Settings
from app.research_context import ResearchContext
from app.state import ResearchPlan, ResearchTask, create_initial_state


class SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(next(self.outputs))


class CriticSummaryTests(unittest.TestCase):
    def settings(self):
        return Settings(_env_file=None, critic_max_attempts=2)

    def state(self):
        state = create_initial_state("预算与现行规范")
        state["research_plan"] = ResearchPlan(tasks=[ResearchTask(
            id=f"task-{number}", question=f"Requested dimension {number}", query=f"query-{number}",
            requirements=[{"id": f"req-{i}", "description": f"Obligation {i}"} for i in range(5)])
            for number in range(5)])
        return state

    def review(self, topics):
        return {"sufficient": False, "completeness": 0, "relevance": 1, "source_quality": 1,
            "missing_topics": topics, "concerns": [], "evidence_issues": [], "new_tasks": [],
            "summary": "Evidence gaps", "requirement_coverage": [
                {"task_id": f"task-{task}", "requirement_id": f"req-{req}", "covered": False,
                 "evidence_ids": [], "reason": "No evidence"} for task in range(5) for req in range(5)]}

    def test_oversized_summary_repaired_with_all_coverage_rows_preserved(self):
        model = SequenceModel([self.review([f"gap-{i}" for i in range(25)]),
                               self.review(["预算金额与分项", "规范准入与留存", "审批要求"])])
        result = evaluate_evidence(self.state(), model, self.settings())
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(result.requirement_coverage), 25)
        feedback = json.loads(model.calls[1]["user"])["validation_feedback"]
        self.assertIn("at most 20", feedback)
        self.assertIn("Preserve EVERY row", feedback)
        self.assertIn("Merge related", feedback)
        payload = json.loads(model.calls[0]["user"])
        self.assertEqual(payload["max_missing_topics"], 20)

    def test_exact_duplicates_removed_without_truncating_distinct_gaps(self):
        draft = EvaluationDraft.model_validate(self.review([" 预算金额 ", "预算金额"] * 30))
        self.assertEqual(draft.missing_topics, ["预算金额"])
        self.assertEqual(len(draft.requirement_coverage), 25)
        with self.assertRaises(ValidationError):
            EvaluationDraft.model_validate(self.review([f"gap-{i}" for i in range(21)]))

    def test_unrepaired_output_still_stops_at_two_attempts(self):
        invalid = self.review([f"gap-{i}" for i in range(21)])
        model = SequenceModel([invalid, invalid])
        with self.assertRaisesRegex(CriticError, "after 2 attempts"):
            evaluate_evidence(self.state(), model, self.settings())
        self.assertEqual(len(model.calls), 2)

    def test_feedback_does_not_echo_rejected_content(self):
        invalid = self.review([f"SECRET-{i}" for i in range(21)])
        invalid["concerns"] = [f"PRIVATE-{i}" for i in range(21)]
        try:
            EvaluationDraft.model_validate(invalid)
        except ValidationError as exc:
            feedback = evaluation_validation_feedback(exc, self.settings())
        self.assertNotIn("SECRET", feedback)
        self.assertNotIn("PRIVATE", feedback)
        self.assertIn("missing_topics", feedback)
        self.assertIn("concerns", feedback)

    def test_planner_receives_scope_rules_and_does_not_require_maximum_tasks(self):
        model = SequenceModel([{"tasks": [
            {"question": "预算内容", "query": "预算", "tool": "retrieve_documents", "requirements": ["预算总额及范围"]},
            {"question": "现行规范", "query": "现行规范", "tool": "retrieve_documents", "requirements": ["文档准入与留存"]}]},
            {"task_reviews": [{"index": index, "keep": True, "reason": "Requested dimension",
                               "requirements": [{"index": 0, "keep": True, "reason": "Requested detail"}]} for index in range(2)]}])
        plan = plan_research("预算与现行规范", model, self.settings(), local_documents_available=True,
            research_context=ResearchContext(scope="document_grounded"),
            local_document_context="Budget, policy, company overview, pilot acceptance documents")
        self.assertEqual(len(plan.tasks), 2)
        prompt = model.calls[0]["system"]
        self.assertIn("ceiling, not a target", prompt)
        self.assertIn("not a request to compare", prompt)
        self.assertIn("do not add fictional-company", prompt)


if __name__ == "__main__":
    unittest.main()
