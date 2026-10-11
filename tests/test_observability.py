"""Metadata redaction, nested graph spans, failures and run isolation."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.graph import run_research
from app.observability import current_recorder, observe_model, span, trace_run, traced_callable
from app.rag.pipeline import build_local_rag
from app.config import PROJECT_ROOT
from examples.graph_demo import DemoGraphModel
from examples.researcher_demo import demo_search


class ObservabilityTests(unittest.TestCase):
    def test_metadata_allowlist_and_jsonl_do_not_contain_content(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            with trace_run(path) as recorder, span("operation", "run") as attributes:
                attributes.update(api_key="SECRET", prompt="PRIVATE PROMPT", user_query="PRIVATE QUESTION",
                                  result_count=2, research_status="sufficient", malicious={"key": "SECRET"})
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("SECRET", text)
            self.assertNotIn("PRIVATE", text)
            self.assertEqual(json.loads(text)["attributes"], {"result_count": 2, "research_status": "sufficient"})
            self.assertEqual(recorder.events[0]["run_id"], recorder.run_id)

    def test_exception_type_recorded_without_message_and_context_restored(self):
        with trace_run() as recorder:
            with self.assertRaisesRegex(RuntimeError, "SECRET"):
                with span("failing", "tool"):
                    raise RuntimeError("SECRET RESPONSE BODY")
            with span("next", "tool"):
                pass
        self.assertEqual(recorder.events[0]["error_type"], "RuntimeError")
        self.assertEqual(recorder.events[0]["status"], "error")
        self.assertIsNone(recorder.events[1]["parent_span_id"])
        self.assertNotIn("SECRET", json.dumps(recorder.events))
        self.assertIsNone(current_recorder())

    def test_graph_nodes_tools_models_share_run_and_parent_ids(self):
        with trace_run() as recorder:
            state = run_research("PRIVATE QUESTION", DemoGraphModel(), Settings(_env_file=None), web_search=demo_search)
        nodes = [event for event in recorder.events if event["kind"] == "node"]
        self.assertEqual([event["name"] for event in nodes], ["planner", "researcher", "researcher", "critic", "writer"])
        node_ids = {event["span_id"] for event in nodes}
        models = [event for event in recorder.events if event["kind"] == "model"]
        self.assertTrue(models)
        self.assertTrue(all(event["parent_span_id"] in node_ids for event in models))
        self.assertEqual({event["run_id"] for event in recorder.events}, {recorder.run_id})
        self.assertEqual(state["report_status"], "generated")
        self.assertNotIn("PRIVATE QUESTION", json.dumps(recorder.events))

    def test_parallel_runs_are_isolated(self):
        def worker(_):
            with trace_run() as recorder, span("parallel", "run"):
                return recorder
        with ThreadPoolExecutor(max_workers=2) as pool:
            recorders = list(pool.map(worker, [1, 2]))
        self.assertNotEqual(recorders[0].run_id, recorders[1].run_id)
        self.assertTrue(all(len(recorder.events) == 1 for recorder in recorders))
        self.assertIsNone(current_recorder())

    def test_trace_file_append_and_nested_run_restore(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            with trace_run(path) as outer:
                with span("outer", "run"):
                    with trace_run(path) as inner, span("inner", "run"):
                        pass
                    self.assertIs(current_recorder(), outer)
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["run_id"], inner.run_id)
            self.assertEqual(events[1]["run_id"], outer.run_id)

    def test_trace_write_failure_does_not_fail_business_operation(self):
        with TemporaryDirectory() as folder:
            with self.assertLogs("deepresearch", level="WARNING"), trace_run(Path(folder)) as recorder:
                result = traced_callable("operation", "tool", lambda: [1, 2])()
            self.assertEqual(result, [1, 2])
            self.assertEqual(len(recorder.events), 1)

    def test_model_wrapper_is_idempotent_and_logs_each_attempt(self):
        model = observe_model(DemoGraphModel())
        self.assertIs(observe_model(model), model)
        with trace_run() as recorder:
            for _ in range(2):
                model.generate(system="SECRET", user="{}", schema={"properties": {"tasks": {"maxItems": 1}}})
        self.assertEqual(len(recorder.events), 2)
        self.assertNotIn("SECRET", json.dumps(recorder.events))

    def test_rag_records_load_chunk_embedding_and_index_metadata(self):
        with trace_run() as recorder:
            rag = build_local_rag([PROJECT_ROOT / "examples" / "doc"], Settings(_env_file=None, embedding_provider="hashing"))
            rag.retrieve_documents("PRIVATE QUERY")
        names = {event["name"] for event in recorder.events}
        self.assertTrue({"rag.load_documents", "rag.chunk_documents", "rag.build_index", "embedding.documents", "embedding.query", "faiss.search"} <= names)
        text = json.dumps(recorder.events)
        self.assertNotIn("PRIVATE QUERY", text)
        self.assertNotIn("18000", text)


if __name__ == "__main__":
    unittest.main()
