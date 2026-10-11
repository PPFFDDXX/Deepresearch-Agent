"""Reference guards for semantic judgments; IDs alone never prove task coverage."""
from app.report_models import ReportDraft, iter_paragraphs
from app.state import AgentState, ReportCoverage, RequirementCoverage
from app.state import AnswerRequirement


class CoverageValidationError(ValueError):
    """Safe, typed diagnostic; arbitrary unknown IDs never enter the feedback text."""
    def __init__(self, code: str, *, task_id: str | None = None, requirement_id: str | None = None,
                 evidence_ids: list[str] | None = None):
        self.code = code
        self.task_id = task_id
        self.requirement_id = requirement_id
        self.evidence_ids = evidence_ids or []
        guidance = {
            "unknown_requirement": "Use only supplied task/requirement IDs.",
            "duplicate_requirement": "Return one row per task/requirement pair.",
            "duplicate_evidence": "Use each evidence ID once within a coverage row.",
            "unknown_evidence": "Unknown evidence IDs were supplied. Copy valid IDs from the evidence inventory.",
            "rejected_evidence": "Coverage cites evidence rejected in evidence_issues. Use other accepted support, or covered=false with evidence_ids=[].",
            "missing_support": "covered=true requires accepted supporting evidence IDs.",
            "uncovered_with_support": "covered=false requires evidence_ids=[].",
            "missing_requirements": "Return every requirement row, including uncovered requirements.",
        }
        location = f" task={task_id} requirement={requirement_id}" if task_id and requirement_id else ""
        super().__init__(f"Coverage error [{code}]{location}: {guidance[code]}")


def requirement_index(state: AgentState) -> dict[tuple[str, str], AnswerRequirement]:
    return {(task.id, item.id): item for task in state["research_plan"].tasks
            for item in task.requirements}


def validate_requirement_coverage(state: AgentState, rows: list[RequirementCoverage],
                                  accepted: set[str], *, require_complete: bool = True) -> None:
    expected = requirement_index(state)
    known_evidence = {item.id for item in state["evidence"]}
    seen = set()
    for row in rows:
        key = (row.task_id, row.requirement_id)
        if key not in expected:
            raise CoverageValidationError("unknown_requirement")
        if key in seen:
            raise CoverageValidationError("duplicate_requirement", task_id=row.task_id, requirement_id=row.requirement_id)
        seen.add(key)
        location = {"task_id": row.task_id, "requirement_id": row.requirement_id}
        if len(set(row.evidence_ids)) != len(row.evidence_ids):
            raise CoverageValidationError("duplicate_evidence", **location)
        if not set(row.evidence_ids) <= known_evidence:
            raise CoverageValidationError("unknown_evidence", **location)
        if not set(row.evidence_ids) <= accepted:
            raise CoverageValidationError("rejected_evidence", **location,
                                          evidence_ids=sorted(set(row.evidence_ids) - accepted))
        if row.covered and not row.evidence_ids:
            raise CoverageValidationError("missing_support", **location)
        if not row.covered and row.evidence_ids:
            raise CoverageValidationError("uncovered_with_support", **location)
    if require_complete and seen != set(expected):
        raise CoverageValidationError("missing_requirements")


def uncovered_requirement_tasks(state: AgentState, rows: list[RequirementCoverage]) -> list[str]:
    covered = {(row.task_id, row.requirement_id) for row in rows if row.covered}
    return [task.id for task in state["research_plan"].tasks
            if any((task.id, item.id) not in covered for item in task.requirements)]


def validate_report_coverage(state: AgentState, draft: ReportDraft, rows: list[ReportCoverage]) -> None:
    """Require every evidence-backed requirement to be answered in cited BODY paragraphs."""
    expected = {(row.task_id, row.requirement_id): set(row.evidence_ids)
                for row in state["critic_assessments"][-1].requirement_coverage if row.covered}
    paragraphs = list(iter_paragraphs(draft))
    seen = set()
    for row in rows:
        key = (row.task_id, row.requirement_id)
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate report coverage key")
        seen.add(key)
        if not row.covered or not row.paragraph_indices:
            raise ValueError(f"Report omitted supported requirement: {key}")
        if len(set(row.paragraph_indices)) != len(row.paragraph_indices):
            raise ValueError("Report paragraph indices must be unique")
        cited = set()
        for index in row.paragraph_indices:
            if type(index) is not int or not 0 <= index < len(paragraphs):
                raise ValueError("Report paragraph index is out of range")
            cited.update(paragraphs[index].evidence_ids)
        if not cited & expected[key]:
            raise ValueError("Report coverage paragraphs must cite mapped supporting evidence")
    if seen != set(expected):
        raise ValueError("Report review must cover every evidence-backed requirement")
