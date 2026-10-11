"""One task per call: select tool, extract grounded candidates, return state delta."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import Field, ValidationError

from app.config import Settings
from app.llm import ModelError, StructuredModel
from app.state import (
    AgentState, Citation, Evidence, Record, ResearchOutcome, ResearchTask,
    SearchResult, StateUpdate, validate_state,
    RetrievalAttempt,
)
from app.tools.models import ResearchMaterial, ToolError, WebSearchResponse
from app.tools.requirement_retrieval import retrieve_for_requirements


class EvidenceDraft(Record):
    material_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class ExtractionDraft(Record):
    evidence: list[EvidenceDraft] = Field(max_length=20)


class ResearcherError(RuntimeError):
    """Invalid execution state or exhausted extraction attempts."""


def record_id(prefix: str, *parts: str) -> str:
    serialized = json.dumps(parts, ensure_ascii=False)
    return prefix + "-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_evidence(
    task: ResearchTask, materials: list[ResearchMaterial], model: StructuredModel, settings: Settings,
) -> list[EvidenceDraft]:
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "researcher.txt").read_text(encoding="utf-8")
    schema = ExtractionDraft.model_json_schema()
    schema["properties"]["evidence"]["maxItems"] = settings.researcher_max_evidence
    indexed = {f"material-{i}": item for i, item in enumerate(materials, start=1)}
    feedback = ""
    for _ in range(settings.researcher_max_attempts):
        try:
            raw = model.generate(system=prompt, schema=schema, user=json.dumps({
                "question": task.question,
                "requirements": [item.model_dump() for item in task.requirements],
                "max_evidence": settings.researcher_max_evidence,
                "materials": [{"material_id": key, **item.model_dump()} for key, item in indexed.items()],
                "validation_feedback": feedback,
            }, ensure_ascii=False))
            drafts = ExtractionDraft.model_validate_json(raw).evidence
            if len(drafts) > settings.researcher_max_evidence:
                raise ValueError("Too many evidence items")
            for item in drafts:
                if item.material_id not in indexed:
                    raise ValueError("Evidence refers to an unknown material_id")
                if item.excerpt not in indexed[item.material_id].text:
                    raise ValueError("Evidence excerpt must be an exact substring of the material")
            return drafts
        except ModelError as exc:
            if not exc.retryable:
                raise ResearcherError(str(exc)) from None
            feedback = str(exc)
        except ValidationError:
            feedback = "Invalid extraction JSON: follow the schema and use nonempty fields"
        except ValueError as exc:
            feedback = str(exc)
    raise ResearcherError(f"Extraction failed after {settings.researcher_max_attempts} attempts: {feedback}")


def build_researcher_node(
    model: StructuredModel, settings: Settings, *,
    web_search: Callable[[str], WebSearchResponse],
    retrieve_documents: Callable[[str], list[ResearchMaterial]] | None = None,
) -> Callable[[AgentState], StateUpdate]:
    def researcher_node(state: AgentState) -> StateUpdate:
        validate_state(state)
        task = state["current_task"]
        if task is None:
            raise ResearcherError("Researcher requires a current_task")
        round_number = state["retry_count"]
        # Refinements receive new IDs. Earlier tasks remain in the plan for provenance,
        # but must not be re-executed just because Critic starts a new round.
        completed = {o.task_id for o in state["research_outcomes"]}

        def next_task() -> ResearchTask | None:
            return next((t for t in state["research_plan"].tasks if t.id not in completed), None)

        if task.id in completed:
            return {"current_task": next_task()}
        update: StateUpdate = {"citations": [], "search_results": [], "evidence": []}
        status = "no_evidence"
        message = "No matching materials were returned"
        retrieval_attempts: list[RetrievalAttempt] = []
        material_limit = settings.search_top_k
        try:
            if task.tool == "web_search":
                response = web_search(task.query)
                materials = [ResearchMaterial(
                    title=hit.title, locator=str(hit.url).split("#", 1)[0], source_type="web",
                    text=hit.snippet, location="search snippet", origin="search_snippet",
                ) for hit in response.results]
            else:
                if retrieve_documents is None:
                    raise ToolError("configuration", "Local retrieval is not configured")
                if task.requirements:
                    material_limit = min(32, max(settings.search_top_k, 2 * len(task.requirements)))
                    materials, retrieval_attempts = retrieve_for_requirements(task, retrieve_documents, limit=material_limit)
                    if not materials and any(item.status == "tool_error" for item in retrieval_attempts):
                        raise ToolError("retrieval_failed", "No usable materials after requirement queries; inspect retrieval_attempts")
                else:
                    materials = retrieve_documents(task.query)
                if any(m.source_type != "local" or m.origin != "local_chunk" for m in materials):
                    raise ToolError("invalid_response", "Local retriever must return local chunks")
            materials = materials[:material_limit]
            existing = {c.id: c for c in state["citations"]}
            sources = []
            for material in materials:
                source_id = record_id("source", material.source_type, material.locator)
                source = existing.get(source_id) or Citation(
                    id=source_id, title=material.title, source_type=material.source_type, locator=material.locator,
                )
                existing[source_id] = source
                sources.append(source)
                update["citations"].append(source)
                update["search_results"].append(SearchResult(
                    id=record_id("result", task.id, source_id, material.text, material.location or ""),
                    task_id=task.id, citation_id=source_id, snippet=material.text,
                ))
            if materials:
                drafts = extract_evidence(task, materials, model, settings)
                for draft in drafts:
                    index = int(draft.material_id.split("-")[1]) - 1
                    material, source = materials[index], sources[index]
                    update["evidence"].append(Evidence(
                        id=record_id("evidence", task.id, source.id, draft.claim, draft.excerpt,
                                     material.location or "", material.origin),
                        task_id=task.id, citation_id=source.id, claim=draft.claim, excerpt=draft.excerpt,
                        location=material.location, origin=material.origin,
                    ))
                status = "success" if drafts else "no_evidence"
                message = "Candidate evidence extracted; sufficiency is not evaluated" if drafts else "No relevant evidence found in materials"
        except ToolError as exc:
            status, message = "tool_error", f"{exc.code}: {exc}"
        except ResearcherError as exc:
            status, message = "extraction_error", str(exc)
        completed.add(task.id)
        update["research_outcomes"] = [ResearchOutcome(
            id=record_id("outcome", task.id, str(round_number)), task_id=task.id,
            round=round_number, status=status, message=message,
            retrieval_attempts=retrieval_attempts,
        )]
        update["current_task"] = next_task()
        return update

    return researcher_node
