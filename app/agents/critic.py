"""Structured evidence review with deterministic guards and bounded refinement."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import Field, ValidationError, field_validator

from app.agents.planner import TaskDraft
from app.agents.refinement import automatic_refinements
from app.plan_scope import ScopeTaskAudit, review_task_scope
from app.config import Settings
from app.llm import ModelError, StructuredModel
from app.coverage import validate_requirement_coverage, uncovered_requirement_tasks, CoverageValidationError
from app.state import (
    AgentState, CriticAssessment, EvidenceIssue, Record, ResearchPlan,
    ResearchTask, StateUpdate, validate_state, RequirementCoverage, ReviewDiagnostic,
)


class RefinementDraft(TaskDraft):
    addresses_task_ids: list[str]


MAX_SUMMARY_ITEMS = 20


class EvaluationDraft(Record):
    sufficient: bool
    completeness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    missing_topics: list[str] = Field(max_length=MAX_SUMMARY_ITEMS)
    concerns: list[str] = Field(max_length=MAX_SUMMARY_ITEMS)
    evidence_issues: list[EvidenceIssue]
    new_tasks: list[RefinementDraft] = Field(max_length=10)
    summary: str = Field(min_length=1)
    requirement_coverage: list[RequirementCoverage] = Field(default_factory=list)
    # Application-owned, excluded from the model schema below.
    scope_audit: list[ScopeTaskAudit] = Field(default_factory=list, exclude=True)

    @field_validator("missing_topics", "concerns", mode="before")
    @classmethod
    def deduplicate_summaries(cls, value: object) -> object:
        """Remove exact repeats only; never truncate or merge distinct semantic gaps."""
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(dict.fromkeys(item.strip() for item in value))
        return value


class CriticError(RuntimeError):
    """Review could not produce a valid structured result."""


def evaluation_validation_feedback(exc: ValidationError, settings: Settings) -> str:
    """Return actionable schema repair guidance without echoing rejected model text."""
    errors = exc.errors(include_input=False, include_url=False)
    details = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['type']}" for error in errors[:5])
    hints = []
    for field in ("missing_topics", "concerns"):
        if any(error["loc"] == (field,) and error["type"] == "too_long" for error in errors):
            hints.append(f"{field} must contain at most {MAX_SUMMARY_ITEMS} items. Merge related items into a few "
                         "task-level summaries; aim for 3-5 only when needed. Do not enumerate each requirement "
                         "or carry forward resolved topics. Preserve EVERY row of requirement_coverage, including "
                         "uncovered requirements; do not truncate that table to shorten the summary.")
    if any(error["loc"] == ("new_tasks",) and error["type"] == "too_long" for error in errors):
        hints.append(f"new_tasks must contain at most {settings.critic_max_tasks} targeted refinements.")
    return f"Invalid evaluation schema ({details}). " + " ".join(hints) + " Return a complete corrected JSON."


def query_key(tool: str, query: str) -> tuple[str, str]:
    """Normalize whitespace/case for exact-query deduplication, not semantics."""
    return tool, " ".join(query.casefold().split())


def uncovered_tasks(state: AgentState, accepted_ids: set[str]) -> list[str]:
    """Evidence on refinements also covers linked ancestors, never sibling tasks."""
    tasks = {task.id: task for task in state["research_plan"].tasks}
    covered = {item.task_id for item in state["evidence"] if item.id in accepted_ids}
    pending = list(covered)
    while pending:
        task_id = pending.pop()
        for parent in tasks[task_id].addresses_task_ids:
            if parent not in covered:
                covered.add(parent)
                pending.append(parent)
    return [task_id for task_id in tasks if task_id not in covered]


def evaluate_evidence(
    state: AgentState, model: StructuredModel, settings: Settings, *,
    local_documents_available: bool = False,
    local_document_context: str = "", web_search_available: bool = True,
    diagnostics: list[ReviewDiagnostic] | None = None,
) -> EvaluationDraft:
    """Ask for semantic review; validate schema, reference IDs and capabilities."""
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "critic.txt").read_text(encoding="utf-8")
    schema = EvaluationDraft.model_json_schema()
    schema["properties"].pop("scope_audit", None)
    schema["properties"]["new_tasks"]["maxItems"] = settings.critic_max_tasks
    if state.get("research_context") and state["research_context"].requirement_checks:
        schema["required"] = [*schema.get("required", []), "requirement_coverage"]
        refinement_schema = schema["$defs"]["RefinementDraft"]
        refinement_schema["required"] = [*refinement_schema.get("required", []), "requirements"]
        refinement_schema["properties"]["requirements"]["minItems"] = 1
    task_ids = {task.id for task in state["research_plan"].tasks}
    evidence_ids = {item.id for item in state["evidence"]}
    payload = {
        "research_context": state["research_context"].model_dump() if state.get("research_context") else None,
        "user_query": state["user_query"],
        "research_plan": state["research_plan"].model_dump(),
        "evidence": [item.model_dump() for item in state["evidence"]],
        "citations": [item.model_dump() for item in state["citations"]],
        "research_outcomes": [item.model_dump() for item in state["research_outcomes"]],
        "previous_assessment": (state["critic_assessments"][-1].model_dump()
                                if state["critic_assessments"] else None),
        "uncovered_task_ids_before_review": uncovered_tasks(state, evidence_ids),
        "available_tools": (["web_search"] if web_search_available else []) + (["retrieve_documents"] if local_documents_available else []),
        "local_document_inventory": local_document_context,
        "round": state["retry_count"],
        "remaining_retries": max(0, settings.max_retries - state["retry_count"]),
        "min_score": settings.critic_min_score,
        "max_new_tasks": settings.critic_max_tasks,
        "max_missing_topics": MAX_SUMMARY_ITEMS,
        "max_concerns": MAX_SUMMARY_ITEMS,
        "valid_evidence_ids": sorted(evidence_ids),
    }
    feedback = ""
    last_error = "Unknown evaluation failure"
    audit = diagnostics if diagnostics is not None else []
    for attempt in range(1, settings.critic_max_attempts + 1):
        try:
            raw = model.generate(system=prompt, user=json.dumps(
                {**payload, "validation_feedback": feedback}, ensure_ascii=False,
            ), schema=schema)
            draft = EvaluationDraft.model_validate_json(raw).model_copy(update={"scope_audit": []})
            if len(draft.new_tasks) > settings.critic_max_tasks:
                raise ValueError("Too many refinement tasks")
            issue_ids = [issue.evidence_id for issue in draft.evidence_issues]
            if len(set(issue_ids)) != len(issue_ids) or not set(issue_ids) <= evidence_ids:
                raise ValueError("Evidence issue IDs must be supplied, unique evidence IDs")
            accepted = evidence_ids - set(issue_ids)
            normalized = []
            for row in draft.requirement_coverage:
                if len(row.evidence_ids) != len(set(row.evidence_ids)):
                    known_key = (row.task_id, row.requirement_id) in {
                        (task.id, requirement.id) for task in state["research_plan"].tasks for requirement in task.requirements}
                    audit.append(ReviewDiagnostic(attempt=attempt, code="duplicate_evidence_normalized",
                        task_id=row.task_id if known_key else None, requirement_id=row.requirement_id if known_key else None,
                        evidence_ids=list(dict.fromkeys(item for item in row.evidence_ids if item in evidence_ids))))
                normalized.append(row.model_copy(update={"evidence_ids": list(dict.fromkeys(row.evidence_ids))}))
            draft = draft.model_copy(update={"requirement_coverage": normalized})
            validate_requirement_coverage(state, draft.requirement_coverage, accepted)
            strict = state.get("research_context") and state["research_context"].requirement_checks
            for task in draft.new_tasks:
                if strict and (not task.requirements or any(not value.strip() for value in task.requirements)
                               or len(set(task.requirements)) != len(task.requirements)):
                    raise ValueError("Refinements must include unique atomic requirements for missing coverage")
                if not set(task.addresses_task_ids) <= task_ids:
                    raise ValueError("Refinement addresses unknown task IDs")
                if len(set(task.addresses_task_ids)) != len(task.addresses_task_ids):
                    raise ValueError("Refinement parent IDs must be unique")
                if task.tool == "retrieve_documents" and not local_documents_available:
                    raise ValueError("Local retrieval is unavailable; use web_search")
                if task.tool == "web_search" and not web_search_available:
                    raise ValueError("Web search is disabled; use retrieve_documents")
            if any(not text.strip() for text in [*draft.missing_topics, *draft.concerns]):
                raise ValueError("Topics and concerns must not be blank")
            if strict and draft.new_tasks:
                filtered, scope_audit = review_task_scope(state["user_query"],
                    [task.model_dump() for task in draft.new_tasks], model, state["research_context"])
                draft = draft.model_copy(update={"new_tasks": [RefinementDraft.model_validate(item) for item in filtered],
                                                 "scope_audit": scope_audit})
            return draft
        except ModelError as exc:
            audit.append(ReviewDiagnostic(attempt=attempt, code="model_error"))
            if not exc.retryable:
                raise CriticError(str(exc)) from None
            last_error = str(exc)
            feedback = "Previous request failed; return a complete evaluation JSON."
        except ValidationError as exc:
            audit.append(ReviewDiagnostic(attempt=attempt, code="schema_error",
                schema_locations=[".".join(map(str, error["loc"])) for error in
                                  exc.errors(include_input=False, include_url=False)[:10]]))
            last_error = feedback = evaluation_validation_feedback(exc, settings)
        except CoverageValidationError as exc:
            audit.append(ReviewDiagnostic(attempt=attempt, code=exc.code, task_id=exc.task_id,
                requirement_id=exc.requirement_id, evidence_ids=exc.evidence_ids))
            last_error = feedback = str(exc)
        except ValueError as exc:
            audit.append(ReviewDiagnostic(attempt=attempt, code="review_validation_error"))
            last_error = feedback = str(exc)
    raise CriticError(f"Evaluation failed after {settings.critic_max_attempts} attempts: {last_error}")


def build_critic_node(
    model: StructuredModel, settings: Settings, *, local_documents_available: bool = False,
    local_document_context: str = "", web_search_available: bool = True,
) -> Callable[[AgentState], StateUpdate]:
    """Return a review node; only this node schedules and increments research retries."""
    def critic_node(state: AgentState) -> StateUpdate:
        validate_state(state)
        if state["research_plan"] is None or state["current_task"] is not None:
            raise ValueError("Critic requires a completed research batch and a plan")
        if state["research_status"] != "running":
            raise ValueError("Critic cannot review a terminated run")
        round_number = state["retry_count"]
        diagnostics: list[ReviewDiagnostic] = []
        if any(item.round == round_number for item in state["critic_assessments"]):
            raise ValueError("Research round already evaluated")
        try:
            draft = evaluate_evidence(state, model, settings,
                                      local_documents_available=local_documents_available,
                                      local_document_context=local_document_context, web_search_available=web_search_available,
                                      diagnostics=diagnostics)
        except CriticError as exc:
            assessment = CriticAssessment(
                id=f"assessment-{round_number}", round=round_number,
                completeness=0, relevance=0, source_quality=0,
                missing_topics=[], concerns=[str(exc)], evidence_issues=[],
                accepted_evidence_ids=[], uncovered_task_ids=uncovered_tasks(state, set()),
                sufficient=False, decision="error", summary="Evaluation unavailable; evidence sufficiency is unknown",
                diagnostics=diagnostics,
            )
            return {"critic_assessments": [assessment], "research_status": "critic_error"}

        rejected = {issue.evidence_id for issue in draft.evidence_issues}
        accepted = [item.id for item in state["evidence"] if item.id not in rejected]
        requirement_tasks = {task.id for task in state["research_plan"].tasks if task.requirements}
        uncovered = [task_id for task_id in uncovered_tasks(state, set(accepted)) if task_id not in requirement_tasks]
        uncovered += uncovered_requirement_tasks(state, draft.requirement_coverage)
        concerns = list(draft.concerns)
        if uncovered:
            concerns.append("No accepted evidence covers tasks: " + ", ".join(uncovered))
        if not accepted:
            concerns.append("No candidate evidence survived this review")
        scores_pass = min(draft.completeness, draft.relevance, draft.source_quality) >= settings.critic_min_score
        sufficient = bool(draft.sufficient and accepted and not uncovered and not draft.missing_topics and scores_pass)
        if draft.sufficient and not sufficient:
            concerns.append("Application guards overrode the model's sufficiency verdict")

        update: StateUpdate = {"current_task": None}
        if sufficient:
            decision, status = "sufficient", "sufficient"
        elif round_number >= settings.max_retries:
            decision, status = "exhausted", "insufficient"
        else:
            seen = {query_key(task.tool, task.query) for task in state["research_plan"].tasks}
            new_tasks = []
            for item in draft.new_tasks:
                key = query_key(item.tool, item.query)
                if key in seen:
                    concerns.append("Skipped duplicate refinement query: " + item.query)
                    continue
                seen.add(key)
                new_tasks.append(ResearchTask(
                    id=f"retry-{round_number + 1}-task-{len(new_tasks) + 1}",
                    question=item.question, query=item.query, tool=item.tool,
                    addresses_task_ids=tuple(item.addresses_task_ids),
                    requirements=[{"id": f"req-{number}", "description": value}
                                  for number, value in enumerate(item.requirements, 1)],
                    refinement_origin="critic",
                ))
            if not new_tasks and draft.requirement_coverage:
                new_tasks = automatic_refinements(state, draft.requirement_coverage,
                    limit=settings.critic_max_tasks, local_available=local_documents_available,
                    web_available=web_search_available)
                if new_tasks:
                    concerns.append("Application generated targeted refinements for uncovered requirements")
            if new_tasks:
                decision, status = "retry", "running"
                update.update(
                    research_plan=ResearchPlan(tasks=[*state["research_plan"].tasks, *new_tasks],
                                              scope_audit=[*state["research_plan"].scope_audit, *draft.scope_audit]),
                    current_task=new_tasks[0], retry_count=round_number + 1,
                )
            else:
                decision, status = "no_new_tasks", "insufficient"
                concerns.append("No novel executable refinement queries were proposed")
        assessment = CriticAssessment(
            id=f"assessment-{round_number}", round=round_number,
            completeness=draft.completeness, relevance=draft.relevance, source_quality=draft.source_quality,
            missing_topics=draft.missing_topics, concerns=concerns, evidence_issues=draft.evidence_issues,
            accepted_evidence_ids=accepted, uncovered_task_ids=uncovered,
            sufficient=sufficient, decision=decision, summary=draft.summary,
            requirement_coverage=draft.requirement_coverage,
            diagnostics=diagnostics,
            scope_audit=draft.scope_audit,
        )
        update.update(critic_assessments=[assessment], research_status=status)
        return update
    return critic_node
