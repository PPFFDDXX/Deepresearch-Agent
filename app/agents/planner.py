"""Structured planning and a node callable compatible with StateGraph."""

import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field, ValidationError

from app.config import Settings
from app.llm import ModelError, StructuredModel
from app.state import AgentState, Record, ResearchPlan, ResearchTask, StateUpdate, validate_state
from app.research_context import ResearchContext
from app.plan_scope import review_task_scope


class TaskDraft(Record):
    question: str = Field(min_length=1)
    query: str = Field(min_length=1)
    tool: Literal["web_search", "retrieve_documents"]
    requirements: list[str] = Field(default_factory=list, max_length=10)


class PlanDraft(Record):
    tasks: list[TaskDraft] = Field(min_length=1, max_length=10)


class PlannerError(RuntimeError):
    """Planning stopped without publishing a partial or invalid plan."""


def plan_research(
    user_query: str, model: StructuredModel, settings: Settings, *, local_documents_available: bool = False,
    local_document_context: str = "", web_search_available: bool = True,
    research_context: ResearchContext | None = None,
) -> ResearchPlan:
    if not user_query.strip():
        raise PlannerError("Research question must not be blank")
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "planner.txt").read_text(encoding="utf-8")
    prompt += f"\nCreate 1 to {settings.planner_max_tasks} tasks. Local documents available: {local_documents_available}."
    prompt += f"\nWeb search available: {web_search_available}."
    if research_context and research_context.requirement_checks:
        prompt += "\nEvery task MUST include 1-10 nonempty requirements: atomic answer obligations, not guessed answers. Use only as many tasks as needed; the maximum is not a quota. Cover original-query dimensions only, not every available document."
    schema = PlanDraft.model_json_schema()
    schema["properties"]["tasks"]["maxItems"] = settings.planner_max_tasks
    if research_context and research_context.requirement_checks:
        task_schema = schema["$defs"]["TaskDraft"]
        task_schema["required"] = [*task_schema.get("required", []), "requirements"]
        task_schema["properties"]["requirements"]["minItems"] = 1
    feedback = ""
    last_error = "Unknown planning failure"
    for attempt in range(settings.planner_max_attempts):
        try:
            raw = model.generate(
                system=prompt,
                user=json.dumps({"research_question": user_query, "local_document_inventory": local_document_context,
                                 "research_context": research_context.model_dump() if research_context else None,
                                 "validation_feedback": feedback}, ensure_ascii=False),
                schema=schema,
            )
            draft = PlanDraft.model_validate_json(raw)
            if len(draft.tasks) > settings.planner_max_tasks:
                raise ValueError("Too many research tasks")
            if research_context and research_context.requirement_checks and any(
                not task.requirements or any(not value.strip() for value in task.requirements)
                or len(set(task.requirements)) != len(task.requirements) for task in draft.tasks
            ):
                raise ValueError("Each task requires unique nonempty atomic answer requirements")
            if not local_documents_available and any(t.tool == "retrieve_documents" for t in draft.tasks):
                raise ValueError("Local retrieval is unavailable; use web_search")
            if not web_search_available and any(t.tool == "web_search" for t in draft.tasks):
                raise ValueError("Web search is disabled; use retrieve_documents")
            normalized = [t.question.casefold() for t in draft.tasks]
            if len(set(normalized)) != len(normalized):
                raise ValueError("Duplicate research questions")
            audit = []
            if research_context and research_context.requirement_checks:
                filtered, audit = review_task_scope(user_query, [task.model_dump() for task in draft.tasks], model, research_context)
                if not filtered:
                    raise ValueError("Scope review removed every task; replan requested dimensions only")
                draft = PlanDraft(tasks=[TaskDraft.model_validate(item) for item in filtered])
            return ResearchPlan(tasks=[
                ResearchTask(id=f"task-{index}", question=task.question, query=task.query, tool=task.tool,
                             requirements=[{"id": f"req-{number}", "description": value}
                                           for number, value in enumerate(task.requirements, 1)])
                for index, task in enumerate(draft.tasks, start=1)
            ], scope_audit=audit)
        except ModelError as exc:
            if not exc.retryable:
                raise PlannerError(str(exc)) from None
            feedback = "Previous request did not complete; return a complete JSON plan."
            last_error = str(exc)
        except ValidationError as exc:
            # Report locations and error types, not raw model output or secret inputs.
            details = "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'root'}: {error['type']}"
                for error in exc.errors(include_input=False, include_url=False)[:5]
            )
            last_error = f"Output validation failed ({details})"
            feedback = "Output failed the schema: use nonempty question/query, allowed tools and the required task list."
            feedback += " " + details
        except ValueError as exc:
            feedback = str(exc)
            last_error = str(exc)
    raise PlannerError(f"Planning failed after {settings.planner_max_attempts} attempts: {last_error}")


def build_planner_node(
    model: StructuredModel, settings: Settings, *, local_documents_available: bool = False,
    local_document_context: str = "", web_search_available: bool = True,
) -> Callable[[AgentState], StateUpdate]:
    def planner_node(state: AgentState) -> StateUpdate:
        validate_state(state)
        if state["research_plan"] is not None:
            raise PlannerError("Initial Planner requires a fresh state; refinement belongs to Critic")
        plan = plan_research(state["user_query"], model, settings,
                             research_context=state.get("research_context"),
                             local_documents_available=local_documents_available,
                             local_document_context=local_document_context, web_search_available=web_search_available)
        return {"research_plan": plan, "current_task": plan.tasks[0]}
    return planner_node
