"""Explicit baseline retrieval, report integrity and rule-based answer checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.citations import approved_evidence, render_report
from app.config import Settings
from app.graph import run_research
from app.llm import ChatCompletionsModel
from app.observability import span
from app.rag.pipeline import build_local_rag
from app.report_models import iter_paragraphs
from app.state import AgentState, validate_state
from app.research_context import build_research_context
from app.tools.models import ToolError, WebSearchResponse


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    retrieval_query: str | None = Field(default=None, min_length=1)
    expected_source: str | None = Field(default=None, min_length=1)
    expected_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    expect_refusal: bool = False
    include_archived: bool = False

    @model_validator(mode="after")
    def valid_expectations(self) -> EvalCase:
        if any(not term.strip() for term in [*self.expected_terms, *self.forbidden_terms]):
            raise ValueError("Evaluation terms must be nonempty")
        if not self.expect_refusal and not self.expected_terms:
            raise ValueError("Answerable cases require expected_terms")
        if not self.expect_refusal and not self.expected_source:
            raise ValueError("Answerable cases require expected_source")
        return self


class EvalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=100)
    cases: list[EvalCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_cases(self) -> EvalDataset:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique")
        if any(not item.strip() for item in self.documents):
            raise ValueError("Document paths must be nonempty")
        return self


def normalize_text(text: str) -> str:
    """Normalize whitespace/case/thousands commas; not semantic equivalence."""
    return re.sub(r"\s+", "", re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text).casefold())


def evaluate_state(state: AgentState) -> dict[str, Any]:
    """Measure structural integrity and heuristic coverage, not factual truth."""
    try:
        validate_state(state)
        latest = state["critic_assessments"][-1] if state["critic_assessments"] else None
        tasks = len(state["research_plan"].tasks) if state["research_plan"] else 0
        outcomes = state["research_outcomes"]
        draft = state["report_draft"]
        paragraphs = list(iter_paragraphs(draft)) if draft else []
        rendering_consistent = None
        if draft is not None:
            rendered, references = render_report(draft, state, approved_evidence(state))
            rendering_consistent = rendered == state["final_report"] and references == state["report_citations"]
        return {
            "state_valid": True, "rendering_consistent": rendering_consistent,
            "task_execution_success_rate": (sum(o.status == "success" for o in outcomes) / len(outcomes) if outcomes else None),
            "critic_task_coverage": ((tasks - len(latest.uncovered_task_ids)) / tasks if tasks and latest else None),
            "candidate_evidence_count": len(state["evidence"]),
            "accepted_evidence_count": len(latest.accepted_evidence_ids) if latest else 0,
            "evidence_issue_count": len(latest.evidence_issues) if latest else 0,
            "report_paragraph_citation_coverage": (sum(bool(p.evidence_ids) for p in paragraphs) / len(paragraphs) if paragraphs else None),
            "report_source_count": len(state["report_citations"]), "retry_count": state["retry_count"],
            "research_status": state["research_status"], "report_status": state["report_status"],
            "semantic_support_verified": False,
            "requirement_checks_enabled": bool(state.get("research_context") and state["research_context"].requirement_checks),
            "report_requirement_coverage": (sum(item.covered for item in state.get("report_coverage", []))
                / len(state["report_coverage"]) if state.get("report_coverage") else None),
        }
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        return {"state_valid": False, "rendering_consistent": False,
                "error_type": type(exc).__name__, "semantic_support_verified": False}


def evaluate_answer(state: AgentState, case: EvalCase) -> dict[str, Any]:
    """Check literal expectations in report BODY, excluding sources and limitations."""
    draft = state["report_draft"]
    body = " ".join(p.text for p in iter_paragraphs(draft)) if draft else ""
    normalized = normalize_text(body)
    expected_found = [term for term in case.expected_terms if normalize_text(term) in normalized]
    forbidden_found = [term for term in case.forbidden_terms if normalize_text(term) in normalized]
    refusal_markers = ("资料不足", "信息不足", "未提供", "无法确定", "无法确认", "不足以", "cannot determine", "insufficient")
    refusal = state["report_status"] == "no_evidence" or any(normalize_text(item) in normalized for item in refusal_markers)
    operational_failure = state["research_status"] == "critic_error" or state["report_status"] == "writer_error"
    if state["report_status"] == "no_evidence" and any(
        outcome.status in ("tool_error", "extraction_error") for outcome in state["research_outcomes"]
    ):
        operational_failure = True
    metrics = evaluate_state(state)
    valid_report = metrics["state_valid"] and metrics["rendering_consistent"] is not False
    terms_pass = len(expected_found) == len(case.expected_terms)
    source_pass = case.expected_source is None or any(
        Path(source.locator).name == case.expected_source for source in state["report_citations"]
    )
    answer_pass = (valid_report and not operational_failure and not forbidden_found
                   and (refusal and terms_pass if case.expect_refusal else terms_pass and source_pass and draft is not None))
    return {"passed": bool(answer_pass), "expected_terms_found": expected_found,
            "forbidden_terms_found": forbidden_found, "refusal_detected": refusal,
            "expected_source_used": source_pass,
            "term_coverage": len(expected_found) / len(case.expected_terms) if case.expected_terms else None,
            "method": "literal_rules_not_semantic_judging", "state_metrics": metrics}


def load_dataset(path: Path) -> EvalDataset:
    return EvalDataset.model_validate_json(path.read_text(encoding="utf-8-sig"))


def disabled_web_search(query: str) -> WebSearchResponse:
    raise ToolError("configuration", "Evaluation is local-only; public search is disabled")


def run_evaluation(path: Path, settings: Settings, *, live: bool = False) -> dict[str, Any]:
    """Offline retrieval by default; explicit live mode adds real embeddings/LLM calls."""
    dataset = load_dataset(path)
    paths = [(path.resolve().parent / item).resolve() for item in dataset.documents]
    effective = settings if live else settings.model_copy(update={"embedding_provider": "hashing"})
    indexes = {}
    results = []
    with span("evaluation.dataset", "evaluation") as attributes:
        attributes["case_count"] = len(dataset.cases)
        for case in dataset.cases:
            with span("evaluation.case", "evaluation") as case_attributes:
                try:
                    if case.include_archived not in indexes:
                        indexes[case.include_archived] = build_local_rag(paths, effective, include_archived=case.include_archived)
                    rag = indexes[case.include_archived]
                    hits = rag.retrieve_documents(case.retrieval_query or case.query)
                    matching = " ".join(hit.text for hit in hits if Path(hit.locator).name == case.expected_source)
                    retrieval_pass = (None if case.expect_refusal else
                                      all(normalize_text(term) in normalize_text(matching) for term in case.expected_terms))
                    item = {"id": case.id, "retrieval_pass": retrieval_pass,
                            "retrieved_files": list(dict.fromkeys(Path(hit.locator).name for hit in hits)),
                            "answer": None, "error_type": None}
                    if live:
                        state = run_research(case.query, ChatCompletionsModel(settings), settings,
                            research_context=build_research_context(rag.documents, local_only=True),
                            web_search=disabled_web_search, retrieve_documents=rag.retrieve_documents,
                            local_document_context=rag.describe_documents(), web_search_available=False,
                            writer_model=ChatCompletionsModel(settings, timeout_seconds=settings.writer_timeout_seconds))
                        item["answer"] = evaluate_answer(state, case)
                    results.append(item)
                    case_attributes["exit_code"] = 0
                except (RuntimeError, ValueError, OSError) as exc:
                    results.append({"id": case.id, "retrieval_pass": False if not case.expect_refusal else None,
                                    "answer": None, "error_type": type(exc).__name__})
                    case_attributes["exit_code"] = 1
                    # Only the type is safe; the exception text may contain input data.
        attributes["result_count"] = len(results)
    retrieval = [item["retrieval_pass"] for item in results if item["retrieval_pass"] is not None]
    answers = [item["answer"]["passed"] if item["answer"] is not None else False for item in results] if live else []
    return {"dataset": dataset.name, "mode": "live" if live else "offline_retrieval",
            "embedding_provider": effective.embedding_provider, "case_count": len(results),
            "error_count": sum(item["error_type"] is not None for item in results),
            "retrieval_pass_rate": sum(retrieval) / len(retrieval) if retrieval else None,
            "answer_pass_rate": sum(answers) / len(answers) if answers else None,
            "semantic_support_verified": False, "cases": results}
