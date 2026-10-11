"""Separate scope review; absence of evidence is not permission to drop a user request."""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.llm import StructuredModel
from app.research_context import ResearchContext


class ScopeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RequirementScope(ScopeRecord):
    index: int = Field(ge=0)
    keep: bool
    reason: str = Field(min_length=1)


class TaskScope(ScopeRecord):
    index: int = Field(ge=0)
    keep: bool
    reason: str = Field(min_length=1)
    requirements: list[RequirementScope]


class ScopeReview(ScopeRecord):
    task_reviews: list[TaskScope]


class ScopeRequirementAudit(ScopeRecord):
    description: str
    kept: bool
    reason: str


class ScopeTaskAudit(ScopeRecord):
    question: str
    kept: bool
    reason: str
    requirements: list[ScopeRequirementAudit]


def review_task_scope(user_query: str, tasks: list[dict], model: StructuredModel,
                      context: ResearchContext) -> tuple[list[dict], list[ScopeTaskAudit]]:
    """Review all indices, prune only explicit irrelevant obligations, preserve audit."""
    prompt = (Path(__file__).parent / "prompts" / "plan_scope.txt").read_text(encoding="utf-8")
    review = ScopeReview.model_validate_json(model.generate(system=prompt,
        user=json.dumps({"user_query": user_query, "research_context": context.model_dump(),
                         "tasks": [{"index": index, **task} for index, task in enumerate(tasks)]}, ensure_ascii=False),
        schema=ScopeReview.model_json_schema()))
    indices = [item.index for item in review.task_reviews]
    if len(set(indices)) != len(indices) or set(indices) != set(range(len(tasks))):
        raise ValueError("Scope review must assess every task index exactly once")
    by_index = {item.index: item for item in review.task_reviews}
    filtered = []
    audit = []
    for index, task in enumerate(tasks):
        decision = by_index[index]
        requirement_indices = [item.index for item in decision.requirements]
        requirements = task["requirements"]
        if len(set(requirement_indices)) != len(requirement_indices) or set(requirement_indices) != set(range(len(requirements))):
            raise ValueError("Scope review must assess every requirement index exactly once")
        indexed = {item.index: item for item in decision.requirements}
        kept_requirements = [text for number, text in enumerate(requirements) if indexed[number].keep]
        if decision.keep and not kept_requirements:
            raise ValueError("A retained task must have at least one relevant requirement")
        audit.append(ScopeTaskAudit(question=task["question"], kept=decision.keep, reason=decision.reason,
            requirements=[ScopeRequirementAudit(description=text, kept=decision.keep and indexed[number].keep,
                                                reason=indexed[number].reason) for number, text in enumerate(requirements)]))
        if decision.keep:
            filtered.append({**task, "requirements": kept_requirements})
    return filtered, audit
