"""Offline integration tests for enforced Critic decisions and refinement rounds."""

import json
import unittest

from app.agents.critic import build_critic_node, query_key, uncovered_tasks
from app.config import Settings
from app.graph import run_research
from app.llm import ModelError
from app.state import (
    Citation, Evidence, ResearchPlan, ResearchTask, create_initial_state, validate_state,
)
from app.tools.models import ResearchMaterial, ToolError, WebSearchResponse
from examples.graph_demo import DEMO_QUERY, DemoGraphModel
from examples.researcher_demo import demo_search


def evaluation(*, sufficient=False, new_tasks=None, issues=None, score=0.9, missing=None):
    return {
        "sufficient": sufficient, "completeness": score, "relevance": score,
        "source_quality": score, "missing_topics": missing or [], "concerns": [],
        "evidence_issues": issues or [], "new_tasks": new_tasks or [], "summary": "Fixture review",
    }


def refinement(query="better primary sources", parents=None, tool="web_search"):
    return {"question": "Find supporting primary evidence", "query": query, "tool": tool,
            "addresses_task_ids": parents if parents is not None else ["task-1"]}


class ReviewModel(DemoGraphModel):
    def __init__(self, reviews):
        self.reviews = iter(reviews)
        self.review_calls = 0
        self.plan_calls = 0

    def generate(self, **kwargs):
        properties = kwargs["schema"]["properties"]
        if "sufficient" in properties:
            self.review_calls += 1
            payload = json.loads(kwargs["user"])
            review = next(self.reviews)
            if isinstance(review, Exception):
                raise review
            if callable(review):
                review = review(payload)
            return review if isinstance(review, str) else json.dumps(review)
        if "tasks" in properties:
            self.plan_calls += 1
        return super().generate(**kwargs)


class CriticTests(unittest.TestCase):
    def settings(self, **kwargs):
        return Settings(_env_file=None, **kwargs)

    def run_graph(self, model, search=demo_search, **settings):
        return run_research(DEMO_QUERY, model, self.settings(**settings), web_search=search)

    def test_sufficient_stops_without_retry(self):
        model = ReviewModel([evaluation(sufficient=True)])
        state = self.run_graph(model)
        self.assertEqual(state["research_status"], "sufficient")
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual(model.review_calls, 1)
        self.assertEqual(len(state["critic_assessments"][-1].accepted_evidence_ids), 2)
        self.assertEqual(state["report_status"], "generated")

    def test_failed_initial_task_is_recovered_without_replaying_old_tasks(self):
        calls = []
        def search(query):
            calls.append(query)
            if query == "fixture metadata":
                raise ToolError("timeout", "Fixture failure")
            return demo_search(query)
        def first_review(payload):
            self.assertEqual(payload["uncovered_task_ids_before_review"], ["task-1"])
            return evaluation(new_tasks=[refinement()])
        model = ReviewModel([first_review, evaluation(sufficient=True)])
        state = self.run_graph(model, search)
        self.assertEqual(calls, ["fixture metadata", "fixture traceability", "better primary sources"])
        self.assertEqual(model.plan_calls, 1)
        self.assertEqual(state["retry_count"], 1)
        self.assertEqual(state["research_status"], "sufficient")
        self.assertEqual([o.round for o in state["research_outcomes"]], [0, 0, 1])
        self.assertEqual(state["research_plan"].tasks[-1].addresses_task_ids, ("task-1",))
        self.assertEqual([a.decision for a in state["critic_assessments"]], ["retry", "sufficient"])
        self.assertEqual(state["critic_assessments"][-1].uncovered_task_ids, [])
        validate_state(state)

    def test_retry_budget_exhausted_keeps_all_rounds(self):
        reviews = [evaluation(new_tasks=[refinement(query=f"query round {i}")]) for i in range(3)]
        model = ReviewModel(reviews)
        state = self.run_graph(model, max_retries=2)
        self.assertEqual(state["retry_count"], 2)
        self.assertEqual(state["research_status"], "insufficient")
        self.assertEqual([a.decision for a in state["critic_assessments"]], ["retry", "retry", "exhausted"])
        self.assertEqual(len(state["research_outcomes"]), 4)
        self.assertEqual(len(state["evidence"]), 4)
        self.assertIsNone(state["current_task"])

    def test_zero_retries_still_evaluates_once(self):
        model = ReviewModel([evaluation(new_tasks=[refinement()])])
        state = self.run_graph(model, max_retries=0)
        self.assertEqual(model.review_calls, 1)
        self.assertEqual(len(state["research_plan"].tasks), 2)
        self.assertEqual(state["critic_assessments"][-1].decision, "exhausted")

    def test_false_sufficient_with_no_evidence_is_overridden(self):
        def empty(query):
            return WebSearchResponse(query=query, provider="fixture", attempts=1, results=[])
        state = self.run_graph(ReviewModel([evaluation(sufficient=True)]), empty)
        self.assertEqual(state["research_status"], "insufficient")
        self.assertFalse(state["critic_assessments"][-1].sufficient)
        self.assertEqual(state["critic_assessments"][-1].uncovered_task_ids, ["task-1", "task-2"])

    def test_low_score_or_missing_topics_override_sufficiency(self):
        for review in (evaluation(sufficient=True, score=0.5),
                       evaluation(sufficient=True, missing=["Latency comparison"])):
            with self.subTest(review=review):
                state = self.run_graph(ReviewModel([review]))
                self.assertEqual(state["research_status"], "insufficient")

    def test_duplicate_queries_stop_without_consuming_a_round(self):
        state = self.run_graph(ReviewModel([evaluation(new_tasks=[refinement("  FIXTURE   METADATA ")])]))
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual(state["critic_assessments"][-1].decision, "no_new_tasks")
        self.assertEqual(query_key("web_search", " a  B "), ("web_search", "a b"))

    def test_batch_deduplicates_and_does_not_mutate_input(self):
        state = create_initial_state("Fixture")
        state["research_plan"] = ResearchPlan(tasks=[ResearchTask(id="task-1", question="Q", query="old")])
        review = evaluation(new_tasks=[refinement(), refinement(" BETTER PRIMARY SOURCES ")])
        node = build_critic_node(ReviewModel([review]), self.settings())
        update = node(state)
        self.assertEqual(len(update["research_plan"].tasks), 2)
        self.assertEqual(len(state["research_plan"].tasks), 1)
        self.assertEqual(state["critic_assessments"], [])
        self.assertEqual(state["retry_count"], 0)

    def test_rejected_evidence_is_retained_but_excluded_from_coverage(self):
        def reject(payload):
            return evaluation(sufficient=True, issues=[{
                "evidence_id": payload["evidence"][0]["id"], "category": "unsupported_claim",
                "reason": "Claim exceeds excerpt",
            }])
        state = self.run_graph(ReviewModel([reject]))
        assessment = state["critic_assessments"][-1]
        self.assertEqual(len(state["evidence"]), 2)
        self.assertEqual(len(assessment.accepted_evidence_ids), 1)
        self.assertEqual(assessment.uncovered_task_ids, ["task-1"])
        self.assertEqual(state["research_status"], "insufficient")

    def test_malformed_review_can_be_repaired(self):
        model = ReviewModel(["invalid JSON", evaluation(sufficient=True)])
        state = self.run_graph(model)
        self.assertEqual(model.review_calls, 2)
        self.assertEqual(state["research_status"], "sufficient")

    def test_unknown_evidence_parents_and_unavailable_tools_fail_closed(self):
        invalid = [
            evaluation(issues=[{"evidence_id": "invented", "category": "weak_source", "reason": "bad"}]),
            evaluation(new_tasks=[refinement(parents=["invented"])]),
            evaluation(new_tasks=[refinement(tool="retrieve_documents")]),
            evaluation(new_tasks=[refinement(query=f"q{i}") for i in range(4)]),
        ]
        for review in invalid:
            with self.subTest(review=review):
                state = self.run_graph(ReviewModel([review, review]))
                self.assertEqual(state["research_status"], "critic_error")
                self.assertEqual(state["critic_assessments"][-1].decision, "error")
                self.assertEqual(len(state["research_outcomes"]), 2)

    def test_model_failures_are_bounded_and_preserve_evidence(self):
        for reviews, expected_calls in (
            ([ModelError("Unavailable", retryable=True)] * 2, 2),
            ([ModelError("Unauthorized", retryable=False)], 1),
        ):
            model = ReviewModel(reviews)
            state = self.run_graph(model)
            self.assertEqual(model.review_calls, expected_calls)
            self.assertEqual(state["research_status"], "critic_error")
            self.assertEqual(len(state["evidence"]), 2)
            self.assertEqual(state["critic_assessments"][-1].accepted_evidence_ids, [])

    def test_transitive_refinement_coverage_and_parent_validation(self):
        state = create_initial_state("Question")
        state["research_plan"] = ResearchPlan(tasks=[
            ResearchTask(id="t1", question="Q1", query="q1"),
            ResearchTask(id="t2", question="Q2", query="q2", addresses_task_ids=("t1",)),
            ResearchTask(id="t3", question="Q3", query="q3", addresses_task_ids=("t2",)),
        ])
        state["citations"] = [Citation(id="s", title="Source", source_type="web", locator="https://example.org")]
        state["evidence"] = [Evidence(id="e", task_id="t3", citation_id="s", claim="Claim", excerpt="Text")]
        validate_state(state)
        self.assertEqual(uncovered_tasks(state, {"e"}), [])
        state["research_plan"] = ResearchPlan(tasks=[ResearchTask(
            id="t1", question="Q1", query="q1", addresses_task_ids=("t1",),
        )])
        with self.assertRaises(ValueError):
            validate_state(state)

    def test_local_refinement_is_selected_when_available(self):
        model = ReviewModel([
            evaluation(new_tasks=[refinement(tool="retrieve_documents")]), evaluation(sufficient=True),
        ])
        calls = []
        def retrieve(query):
            calls.append(query)
            return [ResearchMaterial(title="Local fixture", locator="fixture.txt", source_type="local",
                                     text="Exact local text", origin="local_chunk", location="chunk-1")]
        state = run_research(DEMO_QUERY, model, self.settings(), web_search=demo_search, retrieve_documents=retrieve)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["evidence"][-1].origin, "local_chunk")
        self.assertEqual(state["research_status"], "sufficient")

    def test_largest_configured_loop_fits_graph_step_budget(self):
        class MaximumModel(DemoGraphModel):
            def generate(self, **kwargs):
                props = kwargs["schema"]["properties"]
                if "sufficient" in props:
                    payload = json.loads(kwargs["user"])
                    return json.dumps(evaluation(new_tasks=[refinement(
                        query=f"round {payload['round']} query {i}",
                    ) for i in range(10)]))
                if "tasks" in props:
                    return json.dumps({"tasks": [
                        {"question": f"Q{i}", "query": f"q{i}", "tool": "web_search"} for i in range(10)
                    ]})
                return super().generate(**kwargs)
        state = self.run_graph(MaximumModel(), planner_max_tasks=10, critic_max_tasks=10, max_retries=10)
        self.assertEqual(state["retry_count"], 10)
        self.assertEqual(len(state["research_outcomes"]), 110)
        self.assertEqual(len(state["critic_assessments"]), 11)
        self.assertEqual(state["critic_assessments"][-1].decision, "exhausted")


if __name__ == "__main__":
    unittest.main()
