"""Write only from finally approved candidates; validate before publishing."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import ValidationError, Field

from app.citations import approved_evidence, render_report, render_unavailable_report, validate_report
from app.config import Settings
from app.llm import ModelError, StructuredModel
from app.report_models import ReportDraft
from app.report_models import iter_paragraphs
from app.coverage import validate_report_coverage, requirement_index
from app.state import AgentState, Evidence, StateUpdate, Record, ReportCoverage


class WriterError(RuntimeError):
    """No valid grounded report draft could be obtained within the attempt budget."""


class ReportReview(Record):
    coverage: list[ReportCoverage]
    scope_valid: bool
    scope_reason: str = Field(min_length=1)


def check_report_coverage(state: AgentState, draft: ReportDraft, model: StructuredModel) -> list[ReportCoverage]:
    """Separate semantic review call; application validates IDs, completeness and citations."""
    if not state.get("research_context") or not state["research_context"].requirement_checks:
        return []
    requirements = requirement_index(state)
    supported = [row for row in state["critic_assessments"][-1].requirement_coverage if row.covered]
    payload = {
        "research_context": state["research_context"].model_dump(),
        "requirements": [{**row.model_dump(), "description": requirements[(row.task_id, row.requirement_id)].description}
                         for row in supported],
        "evidence": [item.model_dump() for item in approved_evidence(state)],
        "paragraphs": [{"index": index, **paragraph.model_dump()}
                       for index, paragraph in enumerate(iter_paragraphs(draft))],
    }
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "report_review.txt").read_text(encoding="utf-8")
    review = ReportReview.model_validate_json(model.generate(
        system=prompt, user=json.dumps(payload, ensure_ascii=False), schema=ReportReview.model_json_schema()))
    if not review.scope_valid:
        raise ValueError("Report exceeds source scope: " + review.scope_reason)
    validate_report_coverage(state, draft, review.coverage)
    return review.coverage


def write_report(
    state: AgentState, evidence: Sequence[Evidence], model: StructuredModel, settings: Settings,
    *, review_output: list[ReportCoverage] | None = None,
) -> ReportDraft:
    """Request prose linked to evidence IDs, not model-generated source metadata."""
    if not evidence:
        raise WriterError("Writer requires at least one approved evidence item")
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "writer.txt").read_text(encoding="utf-8")
    assessment = state["critic_assessments"][-1]
    payload = {
        "research_context": state["research_context"].model_dump() if state.get("research_context") else None,
        "requirements": [{"task_id": task.id, **item.model_dump()}
                         for task in state["research_plan"].tasks for item in task.requirements],
        "requirement_coverage": [item.model_dump() for item in assessment.requirement_coverage],
        "user_query": state["user_query"], "research_status": state["research_status"],
        "evidence": [item.model_dump() for item in evidence],
        "missing_topics": assessment.missing_topics,
        "uncovered_tasks": [task.question for task in state["research_plan"].tasks
                            if task.id in assessment.uncovered_task_ids],
        "critic_concerns": assessment.concerns,
    }
    feedback = ""
    last_error = "Unknown writing failure"
    for _ in range(settings.writer_max_attempts):
        try:
            raw = model.generate(system=prompt, user=json.dumps(
                {**payload, "validation_feedback": feedback}, ensure_ascii=False,
            ), schema=ReportDraft.model_json_schema())
            draft = ReportDraft.model_validate_json(raw)
            validate_report(draft, evidence)
            # Check within the existing writing budget; failed drafts are never published.
            coverage = check_report_coverage(state, draft, model)
            if review_output is not None:
                review_output.extend(coverage)
            return draft
        except ModelError as exc:
            if not exc.retryable:
                raise WriterError(str(exc)) from None
            last_error = str(exc)
            feedback = "Previous request failed; return a complete report JSON."
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['type']}"
                for error in exc.errors(include_input=False, include_url=False)[:5]
            )
            last_error = feedback = f"Invalid report schema ({details})"
        except ValueError as exc:
            last_error = feedback = str(exc)
    raise WriterError(f"Writing failed after {settings.writer_max_attempts} attempts: {last_error}")


def build_writer_node(model: StructuredModel, settings: Settings) -> Callable[[AgentState], StateUpdate]:
    """Create a node that publishes a validated report, or an explicit failure report."""
    def writer_node(state: AgentState) -> StateUpdate:
        evidence = approved_evidence(state)
        if state["report_status"] != "not_started":
            raise ValueError("Writer already ran for this state")
        if not evidence:
            return {"final_report": render_unavailable_report(state), "report_status": "no_evidence",
                    "report_draft": None, "report_citations": [], "report_error": None}
        try:
            coverage: list[ReportCoverage] = []
            draft = write_report(state, evidence, model, settings, review_output=coverage)
            markdown, citations = render_report(draft, state, evidence)
        except WriterError as exc:
            return {"final_report": render_unavailable_report(state, writer_error=True),
                    "report_status": "writer_error", "report_error": str(exc),
                    "report_draft": None, "report_citations": []}
        return {"final_report": markdown, "report_draft": draft, "report_citations": citations,
                "report_coverage": coverage,
                "report_status": "generated" if state["research_status"] == "sufficient" else "partial",
                "report_error": None}
    return writer_node
