"""Deterministic grounding checks, citation numbering and Markdown rendering."""

import re
from collections.abc import Sequence
from urllib.parse import quote, urlsplit

from app.report_models import CitedParagraph, ReportCitation, ReportDraft, iter_paragraphs
from app.state import AgentState, Citation, Evidence, validate_state


def approved_evidence(state: AgentState) -> list[Evidence]:
    """Use only the final completed Critic assessment; fail on inconsistent state."""
    validate_state(state)
    if state["research_status"] == "running" or state["current_task"] is not None:
        raise ValueError("Writer requires terminated research")
    if not state["critic_assessments"]:
        raise ValueError("Writer requires a final Critic assessment")
    assessment = state["critic_assessments"][-1]
    expected = {"sufficient": {"sufficient"}, "insufficient": {"exhausted", "no_new_tasks"},
                "critic_error": {"error"}}
    if assessment.round != state["retry_count"] or assessment.decision not in expected[state["research_status"]]:
        raise ValueError("Final assessment does not match research status/round")
    if state["research_status"] == "critic_error":
        return []
    ids = set(assessment.accepted_evidence_ids)
    return [item for item in state["evidence"] if item.id in ids]


def validate_plain_text(text: str) -> None:
    """Model prose may not smuggle its own link, footnote or citation syntax."""
    if re.search(r"[\[\]<>]|https?://|www\.|doi\s*:|arxiv\s*:|file://", text, re.IGNORECASE):
        raise ValueError("Use plain prose and evidence_ids, without links or inline citations")


def validate_report(report: ReportDraft, evidence: Sequence[Evidence]) -> None:
    """Check reference membership; semantic entailment remains a model limitation."""
    allowed = {item.id for item in evidence}
    for paragraph in iter_paragraphs(report):
        validate_plain_text(paragraph.text)
        if not set(paragraph.evidence_ids) <= allowed:
            raise ValueError("Report refers to unknown or rejected evidence IDs")
    for section in report.detailed_analysis:
        validate_plain_text(section.title)


def build_report_citations(
    report: ReportDraft, evidence: Sequence[Evidence], citations: Sequence[Citation],
) -> list[ReportCitation]:
    """Number used sources by first appearance; one number per source across tasks."""
    validate_report(report, evidence)
    indexed_evidence = {item.id: item for item in evidence}
    indexed_sources = {item.id: item for item in citations}
    ordered: dict[str, list[str]] = {}
    for paragraph in iter_paragraphs(report):
        for evidence_id in paragraph.evidence_ids:
            item = indexed_evidence[evidence_id]
            if item.citation_id not in indexed_sources:
                raise ValueError("Report evidence has no source metadata")
            locations = ordered.setdefault(item.citation_id, [])
            if item.location and item.location not in locations:
                locations.append(item.location)
    references = []
    for number, (source_id, locations) in enumerate(ordered.items(), start=1):
        source = indexed_sources[source_id]
        if source.source_type == "web":
            parsed = urlsplit(source.locator)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("Web citations require an HTTP(S) source URL")
        references.append(ReportCitation(
            number=number, citation_id=source.id, title=source.title,
            source_type=source.source_type, locator=source.locator, locations=locations,
        ))
    return references


def escape_markdown(text: str) -> str:
    """Keep source/model text as text, including embedded HTML or Markdown."""
    single_line = " ".join(text.split())
    return re.sub(r"([\\`*_{}\[\]<>#|!+\-])", r"\\\1", single_line)


def render_limitations(state: AgentState, evidence: Sequence[Evidence]) -> str:
    """Render execution limitations from state, outside the model's report claims."""
    assessment = state["critic_assessments"][-1]
    lines = ["## Limitations", "", "- Critic scores and acceptance are heuristic assessments, not fact verification."]
    if state["research_status"] != "sufficient":
        lines.append("- Evidence is incomplete or evaluation failed; this is not a conclusive answer.")
    if any(item.origin == "search_snippet" for item in evidence):
        lines.append("- Search snippets were used; full webpages have not been verified by this workflow.")
    if not evidence:
        lines.append("- No evidence was cleared for report generation.")
    for topic in assessment.missing_topics:
        lines.append("- Missing topic (Critic assessment): " + escape_markdown(topic))
    tasks = {task.id: task for task in state["research_plan"].tasks}
    for task_id in assessment.uncovered_task_ids:
        lines.append("- Uncovered task: " + escape_markdown(tasks[task_id].question))
    from app.coverage import requirement_index
    requirements = requirement_index(state)
    for row in assessment.requirement_coverage:
        if not row.covered:
            lines.append("- Uncovered requirement: " + escape_markdown(requirements[(row.task_id, row.requirement_id)].description))
    for concern in assessment.concerns:
        lines.append("- Critic concern: " + escape_markdown(concern))
    return "\n".join(lines)


def render_report(
    report: ReportDraft, state: AgentState, evidence: Sequence[Evidence],
) -> tuple[str, list[ReportCitation]]:
    """Render a validated draft; application code owns every [N] and source entry."""
    references = build_report_citations(report, evidence, state["citations"])
    numbers = {item.citation_id: item.number for item in references}
    indexed = {item.id: item for item in evidence}

    def paragraph_text(paragraph: CitedParagraph) -> str:
        used = dict.fromkeys(numbers[indexed[item].citation_id] for item in paragraph.evidence_ids)
        return escape_markdown(paragraph.text) + " " + " ".join(f"[{number}]" for number in used)

    lines = ["# Research Report", "", "Research question: " + escape_markdown(state["user_query"]),
             "", "Research status: " + state["research_status"], ""]
    context = state.get("research_context")
    if context and any(item.locator in context.demo_source_locators for item in references):
        lines += ["> 演示资料说明：涉及演示来源的结论仅描述虚构文档中的设定，不证明真实企业预算、支出或产品表现。", ""]
    if state["research_status"] != "sufficient":
        lines += ["> PARTIAL REPORT: evidence gaps remain; findings are provisional.", ""]
    lines += ["## Executive Summary", "", "\n\n".join(map(paragraph_text, report.executive_summary)), "",
              "## Key Findings", "", "\n".join("- " + paragraph_text(p) for p in report.key_findings), "",
              "## Detailed Analysis", ""]
    for section in report.detailed_analysis:
        lines += ["### " + escape_markdown(section.title), "",
                  "\n\n".join(map(paragraph_text, section.paragraphs)), ""]
    lines += ["## Conclusion", "", "\n\n".join(map(paragraph_text, report.conclusion)), "",
              render_limitations(state, evidence), "", "## Sources / Citations", ""]
    for source in references:
        title = escape_markdown(source.title)
        locator = ("[" + title + "](<" + quote(source.locator, safe=":/?#@%=&+;") + ">)"
                   if source.source_type == "web" else title + " — " + escape_markdown(source.locator))
        location = "; ".join(escape_markdown(item) for item in source.locations)
        lines.append(f"[{source.number}] {locator}" + (f" — {location}" if location else ""))
    return "\n".join(lines).rstrip() + "\n", references


def render_unavailable_report(state: AgentState, *, writer_error: bool = False) -> str:
    """No model call and no invented research claims when writing cannot proceed."""
    reason = ("Report generation failed; no unvalidated draft has been published."
              if writer_error else "No evidence was cleared for writing; a grounded answer cannot be generated.")
    lines = ["# Research Report", "", "Research question: " + escape_markdown(state["user_query"]),
             "", "Research status: " + state["research_status"], "", "> REPORT UNAVAILABLE", ""]
    for heading in ("Executive Summary", "Key Findings", "Detailed Analysis", "Conclusion"):
        lines += ["## " + heading, "", reason, ""]
    lines += [render_limitations(state, approved_evidence(state) if writer_error else []), "", "## Sources / Citations", "",
              "No sources cited: no validated report content was published.", ""]
    return "\n".join(lines)
