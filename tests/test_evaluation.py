"""Baseline metrics must distinguish retrieval, structural checks and answer quality."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from app.agents.writer import build_writer_node
from app.config import PROJECT_ROOT, Settings
from app.evaluation import EvalCase, EvalDataset, evaluate_answer, evaluate_state, normalize_text, run_evaluation
from app.graph import run_research
from app.observability import trace_run
from app.state import ResearchOutcome
from examples.graph_demo import DemoGraphModel
from examples.researcher_demo import demo_search
from main import main


def fixture_state():
    return run_research("Fixture", DemoGraphModel(), Settings(_env_file=None), web_search=demo_search)


class EvaluationTests(unittest.TestCase):
    def test_state_metrics_are_structural_not_factual_certification(self):
        metrics = evaluate_state(fixture_state())
        self.assertTrue(metrics["state_valid"])
        self.assertTrue(metrics["rendering_consistent"])
        self.assertEqual(metrics["report_paragraph_citation_coverage"], 1)
        self.assertFalse(metrics["semantic_support_verified"])

    def test_tampered_rendering_and_invalid_source_refs_fail(self):
        state = fixture_state()
        state["final_report"] += "Invented claim [999]"
        self.assertFalse(evaluate_state(state)["rendering_consistent"])
        state["citations"] = []
        self.assertFalse(evaluate_state(state)["state_valid"])

    def test_answer_terms_and_source_must_appear_in_body_not_sources(self):
        state = fixture_state()
        missing = EvalCase(id="missing", query="Q", expected_source="fixture", expected_terms=["not a real source"])
        self.assertFalse(evaluate_answer(state, missing)["passed"])
        good = EvalCase(id="good", query="Q", expected_source="fixture", expected_terms=["citation traceability"])
        self.assertTrue(evaluate_answer(state, good)["passed"])
        wrong_source = good.model_copy(update={"expected_source": "wrong.pdf"})
        self.assertFalse(evaluate_answer(state, wrong_source)["passed"])
        forbidden = good.model_copy(update={"forbidden_terms": ["not real research"]})
        self.assertFalse(evaluate_answer(state, forbidden)["passed"])

    def test_literal_normalization_does_not_claim_semantic_equivalence(self):
        self.assertEqual(normalize_text("18,000 元"), normalize_text("18000元"))
        self.assertNotEqual(normalize_text("1.8万元"), normalize_text("18000元"))

    def test_refusal_does_not_count_critic_or_tool_failure_as_success(self):
        state = fixture_state()
        state.update(report_status="not_started", report_draft=None, report_citations=[], final_report=None,
                     research_status="insufficient")
        state["critic_assessments"] = [state["critic_assessments"][-1].model_copy(update={
            "decision": "exhausted", "sufficient": False, "accepted_evidence_ids": [],
        })]
        state.update(build_writer_node(DemoGraphModel(), Settings(_env_file=None))(state))
        case = EvalCase(id="refusal", query="Unknown", expect_refusal=True)
        self.assertTrue(evaluate_answer(state, case)["passed"])
        state["research_outcomes"] = [ResearchOutcome(id="failed", task_id="task-1", round=0,
                                                     status="tool_error", message="Failed")]
        self.assertFalse(evaluate_answer(state, case)["passed"])

    def test_invalid_and_duplicate_dataset_expectations_rejected(self):
        with self.assertRaises(ValidationError):
            EvalCase(id="no_expectation", query="Q")
        case = {"id": "same", "query": "Q", "expect_refusal": True}
        with self.assertRaises(ValidationError):
            EvalDataset.model_validate({"name": "fixture", "documents": ["docs"], "cases": [case, case]})

    def test_offline_baseline_never_constructs_live_models_and_keeps_failures(self):
        with patch("app.evaluation.ChatCompletionsModel", side_effect=AssertionError("Must remain offline")):
            result = run_evaluation(PROJECT_ROOT / "evals" / "baseline.json", Settings(_env_file=None))
        self.assertEqual(result["embedding_provider"], "hashing")
        self.assertEqual(result["case_count"], 6)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["retrieval_pass_rate"], 0.8)
        self.assertIsNone(result["answer_pass_rate"])
        self.assertIsNone(result["cases"][-1]["retrieval_pass"])
        self.assertFalse(result["cases"][3]["retrieval_pass"])

    def test_dataset_paths_resolve_relative_to_dataset_and_errors_are_safe(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "docs" / "source.txt").write_text("Budget is 18000", encoding="utf-8")
            dataset = root / "eval.json"
            dataset.write_text(json.dumps({"name": "fixture", "documents": ["docs"], "cases": [
                {"id": "budget", "query": "budget", "expected_source": "source.txt", "expected_terms": ["18000"]}
            ]}), encoding="utf-8")
            self.assertEqual(run_evaluation(dataset, Settings(_env_file=None))["retrieval_pass_rate"], 1)
            with patch("app.evaluation.build_local_rag", side_effect=RuntimeError("SECRET INPUT")):
                result = run_evaluation(dataset, Settings(_env_file=None))
            self.assertEqual(result["error_count"], 1)
            self.assertNotIn("SECRET", json.dumps(result))

    def test_cli_evaluation_outputs_json_and_trace_without_real_calls(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "results.json"
            trace = Path(folder) / "trace.jsonl"
            buffer = StringIO()
            with patch("sys.argv", ["main.py", "--evaluate", "--output", str(output), "--trace-file", str(trace)]), \
                    patch("main.load_settings", return_value=Settings(_env_file=None)), redirect_stdout(buffer):
                code = main()
            # The known baseline retrieval gap is an evaluation failure, not an exception.
            self.assertEqual(code, 1)
            data = json.loads(buffer.getvalue())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), data)
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(event["run_id"] == data["run_id"] for event in events))
            self.assertEqual(events[-1]["attributes"]["exit_code"], 1)

    def test_live_evaluation_flow_can_be_tested_with_injected_fixtures(self):
        from app.rag.pipeline import build_local_rag
        rag = build_local_rag([PROJECT_ROOT / "examples" / "doc"], Settings(_env_file=None, embedding_provider="hashing"))
        with patch("app.evaluation.build_local_rag", return_value=rag), \
                patch("app.evaluation.ChatCompletionsModel", return_value=DemoGraphModel()), \
                patch("app.evaluation.run_research", return_value=fixture_state()):
            result = run_evaluation(PROJECT_ROOT / "evals" / "baseline.json", Settings(_env_file=None), live=True)
        self.assertIsNotNone(result["answer_pass_rate"])
        self.assertTrue(all(case["answer"] is not None for case in result["cases"]))

    def test_cli_unexpected_exception_does_not_print_sensitive_message(self):
        buffer = StringIO()
        with patch("sys.argv", ["main.py", "--check-config"]), \
                patch("main.load_settings", return_value=Settings(_env_file=None)), \
                patch("main.execute_action", side_effect=RuntimeError("SECRET API RESPONSE")), redirect_stdout(buffer):
            self.assertEqual(main(), 1)
        self.assertIn("RuntimeError", buffer.getvalue())
        self.assertNotIn("SECRET", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
