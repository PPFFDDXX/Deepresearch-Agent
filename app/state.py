"""Shared research state and validated records for graph nodes."""

from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from app.report_models import ReportCitation, ReportDraft, iter_paragraphs
from app.research_context import ResearchContext
from app.plan_scope import ScopeTaskAudit


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AnswerRequirement(Record):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RequirementCoverage(Record):
    task_id: str
    requirement_id: str
    evidence_ids: list[str]
    covered: bool
    reason: str = Field(min_length=1)


class ReviewDiagnostic(Record):
    attempt: int = Field(ge=1)
    code: str
    task_id: str | None = None
    requirement_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    schema_locations: list[str] = Field(default_factory=list)


class RetrievalAttempt(Record):
    query: str
    result_count: int = Field(ge=0)
    status: Literal["success", "tool_error"]
    error_code: str | None = None


class ReportCoverage(Record):
    task_id: str
    requirement_id: str
    paragraph_indices: list[int]
    covered: bool
    reason: str = Field(min_length=1)


class ResearchTask(Record):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    query: str = Field(min_length=1)
    tool: Literal["web_search", "retrieve_documents"] = "web_search"
    addresses_task_ids: tuple[str, ...] = ()
    requirements: list[AnswerRequirement] = Field(default_factory=list, max_length=10)
    refinement_origin: Literal["planner", "critic", "automatic"] = "planner"

    @model_validator(mode="after")
    def unique_requirements(self) -> "ResearchTask":
        if len({item.id for item in self.requirements}) != len(self.requirements):
            raise ValueError("Requirement IDs must be unique within each task")
        return self


class ResearchPlan(Record):
    tasks: list[ResearchTask] = Field(min_length=1)
    scope_audit: list[ScopeTaskAudit] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_tasks(self) -> "ResearchPlan":
        if len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Task IDs must be unique")
        return self


class Citation(Record):
    """Source metadata; id is an internal identity, not a report footnote number."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_type: Literal["web", "local"]
    locator: str = Field(min_length=1, description="Web URL or local document path")


class SearchResult(Record):
    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    snippet: str = Field(min_length=1)


class Evidence(Record):
    """A claim extracted from material, distinct from the original excerpt."""

    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    location: str | None = Field(default=None, description="Page, section or chunk ID")
    origin: Literal["search_snippet", "local_chunk", "webpage"] = "search_snippet"


class ResearchOutcome(Record):
    """Execution status is distinct from evidence sufficiency."""

    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    round: int = Field(ge=0)
    status: Literal["success", "no_evidence", "tool_error", "extraction_error"]
    message: str = Field(min_length=1)
    retrieval_attempts: list[RetrievalAttempt] = Field(default_factory=list)


class EvidenceIssue(Record):
    """Model-identified concern; this is an assessment, not verified truth."""

    evidence_id: str = Field(min_length=1)
    category: Literal["unsupported_claim", "irrelevant", "weak_source"]
    reason: str = Field(min_length=1)


class CriticAssessment(Record):
    """One evaluation per research round, including the enforced decision."""

    id: str = Field(min_length=1)
    round: int = Field(ge=0)
    completeness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    missing_topics: list[str]
    concerns: list[str]
    evidence_issues: list[EvidenceIssue]
    accepted_evidence_ids: list[str]
    uncovered_task_ids: list[str]
    sufficient: bool
    decision: Literal["sufficient", "retry", "exhausted", "no_new_tasks", "error"]
    summary: str = Field(min_length=1)
    requirement_coverage: list[RequirementCoverage] = Field(default_factory=list)
    diagnostics: list[ReviewDiagnostic] = Field(default_factory=list)
    scope_audit: list[ScopeTaskAudit] = Field(default_factory=list)


T = TypeVar("T", bound=Record)


def merge_records(left: list[T], right: list[T]) -> list[T]:
    """Append new IDs; identical replays are safe, conflicting IDs are errors.

    IDs must be allocated by application code. Nodes return only new records.
    This reducer never modifies its inputs.
    """
    merged: dict[str, T] = {}
    for record in [*left, *right]:
        key = record.id
        if key in merged and merged[key] != record:
            raise ValueError(f"Conflicting record ID: {key}")
        merged[key] = record
    return list(merged.values())


class AgentState(TypedDict):
    research_context: ResearchContext
    report_coverage: list[ReportCoverage]
    user_query: str
    research_plan: ResearchPlan | None
    current_task: ResearchTask | None
    search_results: Annotated[list[SearchResult], merge_records]
    evidence: Annotated[list[Evidence], merge_records]
    citations: Annotated[list[Citation], merge_records]
    retry_count: int
    final_report: str | None
    research_outcomes: Annotated[list[ResearchOutcome], merge_records]
    critic_assessments: Annotated[list[CriticAssessment], merge_records]
    research_status: Literal["running", "sufficient", "insufficient", "critic_error"]
    report_status: Literal["not_started", "generated", "partial", "no_evidence", "writer_error"]
    report_draft: ReportDraft | None
    report_citations: list[ReportCitation]
    report_error: str | None


class StateUpdate(TypedDict, total=False):
    """Nodes return a partial update, rather than mutating their input state."""

    research_context: ResearchContext
    report_coverage: list[ReportCoverage]
    user_query: str
    research_plan: ResearchPlan | None
    current_task: ResearchTask | None
    search_results: list[SearchResult]
    evidence: list[Evidence]
    citations: list[Citation]
    retry_count: int
    final_report: str | None
    research_outcomes: list[ResearchOutcome]
    critic_assessments: list[CriticAssessment]
    research_status: Literal["running", "sufficient", "insufficient", "critic_error"]
    report_status: Literal["not_started", "generated", "partial", "no_evidence", "writer_error"]
    report_draft: ReportDraft | None
    report_citations: list[ReportCitation]
    report_error: str | None


def create_initial_state(user_query: str, research_context: ResearchContext | None = None) -> AgentState:
    query = user_query.strip()
    if not query:
        raise ValueError("user_query must not be blank")
    return AgentState(
        research_context=research_context or ResearchContext(requirement_checks=False), report_coverage=[],
        user_query=query, research_plan=None, current_task=None,
        search_results=[], evidence=[], citations=[], research_outcomes=[], retry_count=0, final_report=None,
        critic_assessments=[], research_status="running",
        report_status="not_started", report_draft=None, report_citations=[], report_error=None,
    )


def validate_state(state: AgentState) -> None:
    """Check cross-record integrity at boundaries; TypedDict does not validate at runtime."""
    if not state["user_query"].strip():
        raise ValueError("user_query must not be blank")
    if type(state["retry_count"]) is not int or state["retry_count"] < 0:
        raise ValueError("retry_count must be a non-negative integer")
    for key in ("search_results", "evidence", "citations", "research_outcomes", "critic_assessments"):
        records = state[key]
        if len({record.id for record in records}) != len(records):
            raise ValueError(f"Duplicate IDs in {key}")
    tasks = {task.id: task for task in state["research_plan"].tasks} if state["research_plan"] else {}
    strict = bool(state.get("research_context") and state["research_context"].requirement_checks)
    if strict and any(not task.requirements for task in tasks.values()):
        raise ValueError("Strict research requires atomic requirements on every task")
    earlier: set[str] = set()
    for task in tasks.values():
        if len(set(task.addresses_task_ids)) != len(task.addresses_task_ids):
            raise ValueError("Duplicate addresses_task_ids")
        if not set(task.addresses_task_ids) <= earlier:
            raise ValueError("Refinement tasks may only address earlier tasks")
        earlier.add(task.id)
    if state["research_status"] not in ("running", "sufficient", "insufficient", "critic_error"):
        raise ValueError("Unknown research_status")
    current = state["current_task"]
    if current is not None and tasks.get(current.id) != current:
        raise ValueError("current_task must belong to research_plan")
    sources = {citation.id for citation in state["citations"]}
    for outcome in state["research_outcomes"]:
        if outcome.task_id not in tasks:
            raise ValueError(f"Unknown outcome task: {outcome.task_id}")
        if outcome.round > state["retry_count"]:
            raise ValueError("Outcome round exceeds retry_count")
    for record in [*state["search_results"], *state["evidence"]]:
        if record.task_id not in tasks:
            raise ValueError(f"Unknown task: {record.task_id}")
        if record.citation_id not in sources:
            raise ValueError(f"Unknown citation: {record.citation_id}")
    evidence_ids = {item.id for item in state["evidence"]}
    for assessment in state["critic_assessments"]:
        for diagnostic in assessment.diagnostics:
            if diagnostic.task_id is not None and diagnostic.task_id not in tasks:
                raise ValueError("Diagnostic refers to unknown task")
            if diagnostic.requirement_id is not None and (diagnostic.task_id not in tasks
                or diagnostic.requirement_id not in {item.id for item in tasks[diagnostic.task_id].requirements}):
                raise ValueError("Diagnostic refers to unknown requirement")
            if not set(diagnostic.evidence_ids) <= evidence_ids:
                raise ValueError("Diagnostic may only expose known evidence IDs")
        if assessment.round > state["retry_count"]:
            raise ValueError("Assessment round exceeds retry_count")
        if not set(assessment.accepted_evidence_ids) <= evidence_ids:
            raise ValueError("Assessment refers to unknown accepted evidence")
        if any(issue.evidence_id not in evidence_ids for issue in assessment.evidence_issues):
            raise ValueError("Assessment issue refers to unknown evidence")
        if not set(assessment.uncovered_task_ids) <= tasks.keys():
            raise ValueError("Assessment refers to unknown task")
        if assessment.sufficient != (assessment.decision == "sufficient"):
            raise ValueError("Assessment sufficiency and decision disagree")
        if set(assessment.accepted_evidence_ids) & {issue.evidence_id for issue in assessment.evidence_issues}:
            raise ValueError("Accepted evidence cannot also be rejected in an assessment")
        from app.coverage import validate_requirement_coverage
        validate_requirement_coverage(state, assessment.requirement_coverage,
                                      set(assessment.accepted_evidence_ids), require_complete=False)
        # A historical assessment predates later refinements. Require only its then-known tasks.
        if strict and assessment.decision != "error":
            reviewed_tasks = {item.task_id for item in state["research_outcomes"] if item.round <= assessment.round}
            expected_keys = {(task.id, item.id) for task in tasks.values() if task.id in reviewed_tasks
                             for item in task.requirements}
            actual_keys = {(row.task_id, row.requirement_id) for row in assessment.requirement_coverage}
            if actual_keys != expected_keys:
                raise ValueError("Assessment requirement coverage does not match its research round")
            if assessment.sufficient and any(not row.covered for row in assessment.requirement_coverage):
                raise ValueError("Sufficient assessment cannot contain uncovered requirements")
    status = state["report_status"]
    if status not in ("not_started", "generated", "partial", "no_evidence", "writer_error"):
        raise ValueError("Unknown report_status")
    if status == "not_started":
        if any((state["final_report"] is not None, state["report_draft"] is not None,
                state["report_citations"], state["report_error"] is not None, state.get("report_coverage", []))):
            raise ValueError("Unstarted Writer cannot have report artifacts")
        return
    if state["research_status"] == "running" or state["current_task"] is not None:
        raise ValueError("A report requires terminated research")
    if not state["critic_assessments"] or not state["final_report"]:
        raise ValueError("A finished Writer requires an assessment and a report")
    if status == "writer_error" and not state["report_error"]:
        raise ValueError("Writer failure must include a safe error reason")
    if status != "writer_error" and state["report_error"] is not None:
        raise ValueError("Successful Writer cannot have report_error")
    if status in ("no_evidence", "writer_error"):
        if state["report_draft"] is not None or state["report_citations"] or state.get("report_coverage", []):
            raise ValueError("Unavailable report cannot publish a draft or citations")
        return
    if state["report_draft"] is None:
        raise ValueError("Generated report requires a structured draft")
    from app.coverage import validate_report_coverage
    if state.get("research_context", ResearchContext(requirement_checks=False)).requirement_checks:
        validate_report_coverage(state, state["report_draft"], state.get("report_coverage", []))
    if (status == "generated" and state["research_status"] != "sufficient"
            or status == "partial" and state["research_status"] != "insufficient"):
        raise ValueError("Report status does not match research status")
    accepted = set(state["critic_assessments"][-1].accepted_evidence_ids)
    used_evidence = {item for paragraph in iter_paragraphs(state["report_draft"]) for item in paragraph.evidence_ids}
    if not used_evidence <= accepted:
        raise ValueError("Report uses evidence not cleared by the final assessment")
    used_sources = {item.citation_id for item in state["evidence"] if item.id in used_evidence}
    references = state["report_citations"]
    if [item.number for item in references] != list(range(1, len(references) + 1)):
        raise ValueError("Report citation numbers must be consecutive")
    if len({item.citation_id for item in references}) != len(references):
        raise ValueError("Report sources must not repeat")
    if {item.citation_id for item in references} != used_sources:
        raise ValueError("Report citations must match sources actually used")
    indexed_sources = {item.id: item for item in state["citations"]}
    for item in references:
        source = indexed_sources[item.citation_id]
        if (item.title, item.locator, item.source_type) != (source.title, source.locator, source.source_type):
            raise ValueError("Report source metadata differs from collected source")
