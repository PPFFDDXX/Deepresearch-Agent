"""Writer/citation integration with genuine graph execution and offline models."""

import copy
import json
import unittest
import httpx
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from app.agents.writer import build_writer_node
from app.citations import approved_evidence, build_report_citations, render_report, validate_report
from app.config import Settings
from app.graph import run_research
from app.llm import ChatCompletionsModel, ModelError
from app.report_models import ReportDraft
from app.state import (
    Citation, CriticAssessment, Evidence, EvidenceIssue, ResearchPlan, ResearchTask,
    create_initial_state, validate_state,
)
from app.tools.models import WebSearchResponse
from examples.graph_demo import DEMO_QUERY, DemoGraphModel
from examples.researcher_demo import demo_search
from main import main


def report_payload(ids, text="Supported fixture finding"):
    paragraph = {"text": text, "evidence_ids": ids}
    return {"executive_summary": [paragraph], "key_findings": [paragraph],
            "detailed_analysis": [{"title": "Fixture analysis", "paragraphs": [paragraph]}],
            "conclusion": [paragraph]}


def reviewed_state(status="sufficient", accepted=None):
    state = create_initial_state("Fixture question")
    state["research_plan"] = ResearchPlan(tasks=[ResearchTask(id="t1", question="Q", query="q")])
    state["citations"] = [
        Citation(id="s1", title="Collected source one", source_type="web", locator="https://example.org/one"),
        Citation(id="s2", title="Collected source two", source_type="local", locator="paper.pdf"),
        Citation(id="unused", title="Unused search source", source_type="web", locator="https://example.org/unused"),
    ]
    state["evidence"] = [
        Evidence(id="e1", task_id="t1", citation_id="s1", claim="C1", excerpt="Text one", location="search snippet"),
        Evidence(id="e2", task_id="t1", citation_id="s1", claim="C2", excerpt="Text two", location="section-2"),
        Evidence(id="e3", task_id="t1", citation_id="s2", claim="C3", excerpt="Text three", origin="local_chunk", location="page 4"),
    ]
    state["research_status"] = status
    state["critic_assessments"] = [CriticAssessment(
        id="assessment-0", round=0, completeness=0.9, relevance=0.9, source_quality=0.9,
        missing_topics=["Fixture gap"] if status == "insufficient" else [], concerns=["Fixture review"],
        evidence_issues=[], accepted_evidence_ids=accepted if accepted is not None else ["e1", "e2", "e3"],
        uncovered_task_ids=[], sufficient=status == "sufficient",
        decision={"sufficient": "sufficient", "insufficient": "exhausted", "critic_error": "error"}[status],
        summary="Fixture assessment",
    )]
    return state


class WriterModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
        self.payloads = []

    def generate(self, **kwargs):
        self.calls += 1
        self.payloads.append(json.loads(kwargs["user"]))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output if isinstance(output, str) else json.dumps(output)


class WriterTests(unittest.TestCase):
    def settings(self):
        return Settings(_env_file=None)

    def test_writer_timeout_override_and_diagnostic(self):
        settings = Settings(_env_file=None, api_key="fixture", llm_timeout_seconds=60)
        def slow(request):
            self.assertEqual(request.extensions["timeout"]["read"], 180)
            self.assertEqual(request.extensions["timeout"]["connect"], 10)
            raise httpx.ReadTimeout("Fixture timeout", request=request)
        model = ChatCompletionsModel(settings, timeout_seconds=180, transport=httpx.MockTransport(slow))
        with self.assertRaisesRegex(ModelError, "ReadTimeout; read timeout=180s") as error:
            model.generate(system="Fixture", user="Fixture", schema={})
        self.assertTrue(error.exception.retryable)
        self.assertEqual(ChatCompletionsModel(settings).timeout_seconds, 60)

    def test_graph_uses_separate_writer_model(self):
        class ResearchOnlyModel(DemoGraphModel):
            def generate(self, **kwargs):
                if "executive_summary" in kwargs["schema"]["properties"]:
                    raise AssertionError("Research model must not write")
                return super().generate(**kwargs)
        class SeparateWriter:
            def generate(self, **kwargs):
                evidence_id = json.loads(kwargs["user"])["evidence"][0]["id"]
                return json.dumps(report_payload([evidence_id]))
        state = run_research(DEMO_QUERY, ResearchOnlyModel(), self.settings(),
                             web_search=demo_search, writer_model=SeparateWriter())
        self.assertEqual(state["report_status"], "generated")

    def test_graph_generates_all_sections_and_existing_source(self):
        state = run_research(DEMO_QUERY, DemoGraphModel(), self.settings(), web_search=demo_search)
        validate_state(state)
        self.assertEqual(state["report_status"], "generated")
        for heading in ("Executive Summary", "Key Findings", "Detailed Analysis", "Conclusion", "Sources / Citations"):
            self.assertIn("## " + heading, state["final_report"])
        self.assertEqual(len(state["report_citations"]), 1)
        self.assertEqual(state["report_citations"][0].locator, state["citations"][0].locator)
        self.assertIn("Search snippets were used", state["final_report"])

    def test_first_use_numbering_shared_source_and_local_locations(self):
        state = reviewed_state()
        payload = report_payload(["e3", "e1", "e2"])
        draft = ReportDraft.model_validate(payload)
        evidence = approved_evidence(state)
        markdown, references = render_report(draft, state, evidence)
        self.assertEqual([item.citation_id for item in references], ["s2", "s1"])
        self.assertEqual(references[0].locations, ["page 4"])
        self.assertEqual(references[1].locations, ["search snippet", "section-2"])
        self.assertIn("Supported fixture finding [1] [2]", markdown)
        self.assertNotIn("[1] [2] [2]", markdown)
        self.assertNotIn("Unused search source", markdown)
        self.assertIn("paper.pdf", markdown)

    def test_only_final_accepted_evidence_is_sent_to_model(self):
        state = reviewed_state(accepted=["e1"])
        state["critic_assessments"][0] = state["critic_assessments"][0].model_copy(update={
            "evidence_issues": [EvidenceIssue(evidence_id="e2", category="unsupported_claim", reason="Fixture issue")],
        })
        before = copy.deepcopy(state)
        model = WriterModel([report_payload(["e1"])])
        update = build_writer_node(model, self.settings())(state)
        self.assertEqual([item["id"] for item in model.payloads[0]["evidence"]], ["e1"])
        self.assertEqual(state, before)
        self.assertEqual(update["report_status"], "generated")

    def test_rejected_or_unknown_ids_are_never_published(self):
        for invalid_id in ("e2", "fabricated"):
            with self.subTest(invalid_id=invalid_id):
                state = reviewed_state(accepted=["e1"])
                model = WriterModel([report_payload([invalid_id])] * 2)
                update = build_writer_node(model, self.settings())(state)
                self.assertEqual(model.calls, 2)
                self.assertEqual(update["report_status"], "writer_error")
                self.assertIsNone(update["report_draft"])
                self.assertEqual(update["report_citations"], [])
                self.assertNotIn("Supported fixture finding", update["final_report"])

    def test_manual_citations_links_and_source_fields_rejected(self):
        state = reviewed_state()
        for text in ("Claim [99]", "See https://invented.example/paper", "Claim <script>", "Reference doi:10.123/example"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    validate_report(ReportDraft.model_validate(report_payload(["e1"], text)), state["evidence"])
        payload = report_payload(["e1"])
        payload["sources"] = [{"url": "https://invented.example"}]
        with self.assertRaises(ValidationError):
            ReportDraft.model_validate(payload)
        payload = report_payload(["e1"])
        payload["detailed_analysis"][0]["title"] = "See [99]"
        with self.assertRaises(ValueError):
            validate_report(ReportDraft.model_validate(payload), state["evidence"])

    def test_missing_sections_uncited_and_duplicate_refs_rejected(self):
        for change in ("missing_section", "uncited", "duplicate"):
            payload = report_payload(["e1"])
            if change == "missing_section":
                del payload["conclusion"]
            elif change == "uncited":
                payload["key_findings"] = [{"text": "No reference", "evidence_ids": []}]
            else:
                payload["conclusion"] = [{"text": "Duplicate", "evidence_ids": ["e1", "e1"]}]
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ReportDraft.model_validate(payload)

    def test_invalid_json_and_grounding_can_be_repaired(self):
        state = reviewed_state()
        for bad in ("not JSON", report_payload(["invented"])):
            model = WriterModel([bad, report_payload(["e1"])])
            update = build_writer_node(model, self.settings())(state)
            self.assertEqual(update["report_status"], "generated")
            self.assertEqual(model.calls, 2)
            self.assertTrue(model.payloads[1]["validation_feedback"])

    def test_writer_model_failures_are_bounded_and_preserve_research(self):
        for outputs, calls in (([ModelError("Unavailable", retryable=True)] * 2, 2),
                               ([ModelError("Unauthorized")], 1)):
            state = reviewed_state()
            model = WriterModel(outputs)
            update = build_writer_node(model, self.settings())(state)
            final = {**state, **update}
            validate_state(final)
            self.assertEqual(model.calls, calls)
            self.assertEqual(final["research_status"], "sufficient")
            self.assertEqual(final["report_status"], "writer_error")
            self.assertEqual(len(final["evidence"]), 3)
            self.assertNotIn("No evidence was cleared for report generation", final["final_report"])

    def test_insufficient_evidence_produces_explicit_partial_report(self):
        state = reviewed_state("insufficient", accepted=["e1"])
        update = build_writer_node(WriterModel([report_payload(["e1"])]), self.settings())(state)
        validate_state({**state, **update})
        self.assertEqual(update["report_status"], "partial")
        self.assertIn("PARTIAL REPORT", update["final_report"])
        self.assertIn("Fixture gap", update["final_report"])

    def test_no_evidence_or_critic_error_skips_model(self):
        for status in ("insufficient", "critic_error"):
            state = reviewed_state(status, accepted=[])
            model = WriterModel([])
            update = build_writer_node(model, self.settings())(state)
            validate_state({**state, **update})
            self.assertEqual(model.calls, 0)
            self.assertEqual(update["report_status"], "no_evidence")
            self.assertEqual(update["report_citations"], [])
            self.assertIn("REPORT UNAVAILABLE", update["final_report"])

    def test_no_search_results_finishes_graph_without_writer_call(self):
        def empty(query):
            return WebSearchResponse(query=query, provider="fixture", attempts=1, results=[])
        state = run_research(DEMO_QUERY, DemoGraphModel(), self.settings(), web_search=empty)
        self.assertEqual(state["report_status"], "no_evidence")
        self.assertIn("No sources cited", state["final_report"])

    def test_source_text_and_link_destination_are_safely_rendered(self):
        state = reviewed_state()
        state["citations"][0] = state["citations"][0].model_copy(update={
            "title": "[False link](javascript:evil) <script> *title*",
            "locator": "https://example.org/a(b)?query=<unsafe>",
        })
        markdown, references = render_report(ReportDraft.model_validate(report_payload(["e1"])), state, approved_evidence(state))
        self.assertNotIn("<script>", markdown)
        self.assertNotIn("[False link](javascript:evil)", markdown)
        self.assertIn("https://example.org/a%28b%29?query=%3Cunsafe%3E", markdown)
        self.assertEqual(references[0].title, state["citations"][0].title)

    def test_missing_source_or_unsafe_web_scheme_is_rejected(self):
        state = reviewed_state()
        draft = ReportDraft.model_validate(report_payload(["e1"]))
        with self.assertRaises(ValueError):
            build_report_citations(draft, state["evidence"], [])
        sources = [state["citations"][0].model_copy(update={"locator": "javascript:evil"})]
        with self.assertRaises(ValueError):
            build_report_citations(draft, state["evidence"], sources)

    def test_writer_requires_final_assessment_and_cannot_run_twice(self):
        state = reviewed_state()
        state["research_status"] = "running"
        with self.assertRaises(ValueError):
            approved_evidence(state)
        state = reviewed_state()
        state["retry_count"] = 1
        with self.assertRaises(ValueError):
            approved_evidence(state)
        state = reviewed_state()
        node = build_writer_node(WriterModel([report_payload(["e1"])]), self.settings())
        final = {**state, **node(state)}
        with self.assertRaises(ValueError):
            node(final)

    def test_state_rejects_tampered_report_metadata(self):
        state = reviewed_state()
        final = {**state, **build_writer_node(WriterModel([report_payload(["e1"])]), self.settings())(state)}
        for change in ({"number": 9}, {"locator": "https://invented.example"}):
            altered = {**final, "report_citations": [final["report_citations"][0].model_copy(update=change)]}
            with self.assertRaises(ValueError):
                validate_state(altered)

    def test_cli_prints_and_saves_utf8_markdown(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "nested" / "report.md"
            output = StringIO()
            with patch("sys.argv", ["main.py", "--demo-graph", "--report-only", "--output", str(target)]), \
                    patch("main.load_settings", return_value=self.settings()), redirect_stdout(output):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("## Executive Summary", target.read_text(encoding="utf-8"))
            self.assertIn("# Research Report", output.getvalue())


if __name__ == "__main__":
    unittest.main()
