"""Reproducible OFFLINE fixtures: real docs/FAISS/graph, deterministic model outputs."""
import argparse
import json
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings
from app.citations import escape_markdown
from app.evaluation import evaluate_state
from app.graph import run_research
from app.rag.pipeline import build_local_rag
from app.research_context import build_research_context
from app.service import serialize_state
from app.state import AgentState, validate_state


CASES = {
    "success": ("根据虚构演示资料，说明试点预算上限、费用分项及预算是否等于实际支出。",
                ["试点预算上限 人民币", "预算分项 模型调用 检索服务 文档清理 预备金", "预算不是实际支出"]),
    "partial": ("根据虚构演示资料，说明试点预算上限和实际支出金额。",
                ["试点预算上限 人民币", "试点实际支出金额"]),
    "refusal": ("根据虚构演示资料，星澜科技的融资金额是多少？资料未提供时明确说明。",
                ["星澜科技融资金额"]),
}
EXPECTED = {"success": ("sufficient", "generated"), "partial": ("insufficient", "partial"),
            "refusal": ("insufficient", "no_evidence")}


def supports(description: str, excerpt: str) -> bool:
    """Golden fixture rules, NOT a semantic evaluator or general research policy."""
    if description == "试点预算上限 人民币":
        return "预算上限为人民币" in excerpt
    if description == "预算分项 模型调用 检索服务 文档清理 预备金":
        return all(term in excerpt for term in ("模型调用预算", "检索服务预算", "文档清理预算", "预备金"))
    if description == "预算不是实际支出":
        return "预算不是实际支出" in excerpt
    return False


class ShowcaseModel:
    def __init__(self, case: str):
        self.case = case

    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = json.loads(user)
        properties = schema["properties"]
        if "tasks" in properties:
            output = {"tasks": [{"question": CASES[self.case][0], "query": CASES[self.case][1][0],
                "tool": "retrieve_documents", "requirements": CASES[self.case][1]}]}
        elif "task_reviews" in properties:
            output = {"task_reviews": [{"index": task["index"], "keep": True,
                "reason": "Fixed showcase question explicitly requests this topic",
                "requirements": [{"index": index, "keep": True, "reason": "Explicit showcase requirement"}
                                 for index in range(len(task["requirements"]))]} for task in payload["tasks"]]}
        elif "sufficient" in properties:
            rows = []
            for task in payload["research_plan"]["tasks"]:
                for requirement in task["requirements"]:
                    ids = [item["id"] for item in payload["evidence"] if supports(requirement["description"], item["excerpt"])]
                    rows.append({"task_id": task["id"], "requirement_id": requirement["id"], "evidence_ids": ids[:1],
                                 "covered": bool(ids), "reason": "Golden passage found" if ids else "Requested amount not provided"})
            output = {"sufficient": all(row["covered"] for row in rows), "completeness": 1 if self.case == "success" else 0.5,
                "relevance": 1, "source_quality": 1, "missing_topics": [] if self.case == "success" else ["请求金额未提供"],
                "concerns": ["OFFLINE FIXTURE: scores and judgments are scripted, not model-quality results"],
                "evidence_issues": [], "new_tasks": [], "summary": "Deterministic showcase", "requirement_coverage": rows}
        elif "executive_summary" in properties:
            selected = {}
            for item in payload["evidence"]:
                selected.setdefault(item["excerpt"], {"text": "以下为虚构演示文档的设定：" + item["claim"], "evidence_ids": [item["id"]]})
            paragraphs = list(selected.values())[:4]
            output = {"executive_summary": paragraphs, "key_findings": paragraphs,
                "detailed_analysis": [{"title": "虚构资料中的原文信息", "paragraphs": paragraphs}], "conclusion": paragraphs}
        elif "scope_valid" in properties:
            output = {"scope_valid": True, "scope_reason": "Scripted demo attribution check", "coverage": [
                {"task_id": row["task_id"], "requirement_id": row["requirement_id"], "covered": True,
                 "paragraph_indices": [paragraph["index"] for paragraph in payload["paragraphs"]
                    if set(paragraph["evidence_ids"]) & set(row["evidence_ids"])], "reason": "Golden fixture paragraph"}
                for row in payload["requirements"]]}
        else:
            selected = []
            for item in payload["materials"]:
                if self.case != "refusal" and item["locator"].endswith("04-pilot-budget.txt") and any(
                    token in item["text"] for token in ("预算上限为人民币", "模型调用预算", "本文没有记录任何实际支出")
                ):
                    selected.append({"material_id": item["material_id"], "claim": item["text"], "excerpt": item["text"]})
            output = {"evidence": selected[:payload["max_evidence"]]}
        return json.dumps(output, ensure_ascii=False)


def run_case(case: str) -> AgentState:
    # Explicit defaults take precedence over DR_* environment values, so the
    # showcase cannot inherit a live provider, a trace path or altered budgets.
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    defaults.update(api_key=None, tavily_api_key=None, embedding_api_key=None,
                    embedding_provider="hashing", trace_file=None)
    settings = Settings(_env_file=None, **defaults)
    rag = build_local_rag([PROJECT_ROOT / "examples/doc"], settings)
    def disabled_search(query: str):
        raise AssertionError("Offline showcase must never call Web Search")
    state = run_research(CASES[case][0], ShowcaseModel(case), settings, web_search=disabled_search,
        retrieve_documents=rag.retrieve_documents, local_document_context=rag.describe_documents(),
        web_search_available=False, research_context=build_research_context(rag.documents, local_only=True))
    validate_state(state)
    if (state["research_status"], state["report_status"]) != EXPECTED[case]:
        raise RuntimeError(f"Showcase regression: {case} returned an unexpected terminal state")
    return state


def portable_value(value: Any) -> Any:
    """Replace workstation paths consistently in public report and audit snapshots."""
    if isinstance(value, str):
        return value.replace(escape_markdown(str(PROJECT_ROOT)), "PROJECT_ROOT").replace(str(PROJECT_ROOT), "PROJECT_ROOT")
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    return value


def export_showcase(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": "offline_fixture", "model": "scripted", "embedding": "lexical_hashing",
                "semantic_support_verified": False, "cases": []}
    for case in CASES:
        state = run_case(case)
        serialized = portable_value({"mode": "offline_fixture", **serialize_state(state), "evaluation": evaluate_state(state)})
        text = json.dumps(serialized, ensure_ascii=False, indent=2)
        report = portable_value(state["final_report"])
        (output / f"{case}.json").write_text(text + "\n", encoding="utf-8")
        (output / f"{case}.md").write_text(
            "> OFFLINE FIXTURE：真实样例文档、FAISS 与 LangGraph；模型判断固定，不代表真实 API 质量。\n\n" + report,
            encoding="utf-8")
        manifest["cases"].append({"id": case, "research_status": state["research_status"],
            "report_status": state["report_status"], "retry_count": state["retry_count"],
            "report": f"{case}.md", "audit": f"{case}.json"})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export three deterministic offline showcase cases")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/showcase")
    args = parser.parse_args()
    print(json.dumps(export_showcase(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
