"""Real local FAISS retrieval with fixture Agent outputs; no LLM/search API calls."""

import json
from typing import Any

from app.config import Settings
from app.graph import run_research
from app.rag.pipeline import LocalRag
from examples.graph_demo import DemoGraphModel


class DemoRagModel(DemoGraphModel):
    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        props = schema["properties"]
        if "tasks" in props:
            return json.dumps({"tasks": [
                {"question": "知舟试点的预算上限与费用分项是什么？", "query": "试点预算上限 预算分项 模型调用", "tool": "retrieve_documents"},
                {"question": "现行知识库复核周期和审计日志留存规定是什么？", "query": "现行知识库 复核 审计日志 更新留存", "tool": "retrieve_documents"},
            ][:props["tasks"]["maxItems"]]}, ensure_ascii=False)
        if "evidence" in props:
            materials = json.loads(user)["materials"]
            return json.dumps({"evidence": [
                {"material_id": item["material_id"], "claim": item["text"], "excerpt": item["text"]}
                for item in materials[:props["evidence"]["maxItems"]]
            ]}, ensure_ascii=False)
        if "executive_summary" in props:
            selected = []
            seen = set()
            for item in json.loads(user)["evidence"]:
                if item["task_id"] not in seen:
                    seen.add(item["task_id"])
                    selected.append({"text": item["claim"], "evidence_ids": [item["id"]]})
            return json.dumps({"executive_summary": selected, "key_findings": selected,
                "detailed_analysis": [{"title": "虚构演示文档的实际检索片段", "paragraphs": selected}],
                "conclusion": selected}, ensure_ascii=False)
        return super().generate(system=system, user=user, schema=schema)


def run_demo(settings: Settings, rag: LocalRag) -> dict[str, Any]:
    def disabled_search(query: str):
        raise AssertionError("Offline local demo must never call Web Search")
    state = run_research("根据本地资料说明知舟试点预算和现行知识库留存规定", DemoRagModel(), settings,
                         web_search=disabled_search, retrieve_documents=rag.retrieve_documents,
                         local_document_context=rag.describe_documents(), web_search_available=False)
    return {key: ([item.model_dump() for item in value] if isinstance(value, list)
                  else value.model_dump() if hasattr(value, "model_dump") else value)
            for key, value in state.items()}
