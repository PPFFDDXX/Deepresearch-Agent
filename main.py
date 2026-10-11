"""Phase 11 CLI entry points: research, metadata-only tracing and evaluation."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from pydantic import ValidationError

from app.agents.planner import build_planner_node
from app.config import PROJECT_ROOT, Settings, load_settings
from app.graph import run_research
from app.llm import ChatCompletionsModel
from app.observability import configure_logging, observe_model, span, trace_run, traced_callable
from app.state import create_initial_state
from app.service import build_live_search, serialize_state
from app.research_context import build_research_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepResearch Agent - Phase 11")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-config", action="store_true")
    for name in ("plan", "search", "research", "retrieve"):
        actions.add_argument("--" + name, metavar="QUERY")
    for name in ("demo-graph", "demo-critic", "demo-rag", "demo-research"):
        actions.add_argument("--" + name, action="store_true")
    actions.add_argument("--evaluate", nargs="?", type=Path,
                         const=PROJECT_ROOT / "evals" / "baseline.json", metavar="DATASET")
    parser.add_argument("--eval-live", action="store_true", help="Real LLM calls for --evaluate")
    parser.add_argument("--documents", nargs="+", type=Path)
    parser.add_argument("--include-archived", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Research Markdown or evaluation JSON")
    parser.add_argument("--trace-file", type=Path, help="Append metadata-only traces as JSONL")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def emit_report(output: dict, args: argparse.Namespace) -> int:
    """Share display/export behavior across live research and retained offline demos."""
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output["final_report"], encoding="utf-8")
    print(output["final_report"] if args.report_only else json.dumps(output, ensure_ascii=False, indent=2))
    if output["report_status"] == "writer_error" or output["research_status"] == "critic_error":
        return 1
    if output["research_status"] == "insufficient":
        import sys
        print("Research stopped with insufficient evidence. See the final critic assessment.", file=sys.stderr)
    return 0


def execute_action(args: argparse.Namespace, settings: Settings, run_id: str) -> int:
    if args.demo_research:
        from examples.researcher_demo import run_demo
        print("OFFLINE FIXTURE: synthetic evidence; no API calls or real research.")
        print(json.dumps(run_demo(settings), ensure_ascii=False, indent=2))
        return 0
    if args.demo_graph or args.demo_critic or args.demo_rag:
        if args.demo_rag:
            from app.rag.pipeline import build_local_rag
            from examples.rag_demo import run_demo
            rag = build_local_rag(args.documents or [PROJECT_ROOT / "examples" / "doc"],
                settings.model_copy(update={"embedding_provider": "hashing"}), include_archived=args.include_archived)
            output = run_demo(settings, rag)
        else:
            if args.demo_critic:
                from examples.critic_demo import run_demo
            else:
                from examples.graph_demo import run_demo
            output = run_demo(settings)
        print("OFFLINE FIXTURE: synthetic evidence; no API calls or real research.")
        return emit_report({"run_id": run_id, **output}, args)
    if args.evaluate is not None:
        from app.evaluation import run_evaluation
        result = {"run_id": run_id, **run_evaluation(args.evaluate, settings, live=args.eval_live)}
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        failed = (result["error_count"]
                  or result["retrieval_pass_rate"] is not None and result["retrieval_pass_rate"] < 1
                  or args.eval_live and result["answer_pass_rate"] < 1)
        return int(bool(failed))
    rag = None
    if args.documents:
        from app.rag.pipeline import build_local_rag
        rag = build_local_rag(args.documents, settings, include_archived=args.include_archived)
    if args.retrieve is not None:
        results = traced_callable("retrieve_documents", "tool", rag.retrieve_documents)(args.retrieve)
        print(json.dumps({"run_id": run_id, "documents": len(rag.documents), "chunks": len(rag.store.chunks),
                          "excluded_archived": rag.excluded_archived,
                          "results": [item.model_dump() for item in results]}, ensure_ascii=False, indent=2))
        return 0
    if args.research is not None:
        from app.evaluation import evaluate_state
        state = run_research(args.research, ChatCompletionsModel(settings), settings,
            research_context=build_research_context(rag.documents if rag else (), local_only=args.local_only),
            web_search=build_live_search(settings, enabled=not args.local_only),
            retrieve_documents=rag.retrieve_documents if rag else None,
            local_document_context=rag.describe_documents() if rag else "", web_search_available=not args.local_only,
            writer_model=ChatCompletionsModel(settings, timeout_seconds=settings.writer_timeout_seconds))
        output = {"run_id": run_id, **serialize_state(state), "evaluation": evaluate_state(state)}
        return emit_report(output, args)
    if args.plan is not None:
        node = traced_callable("planner", "node",
                              build_planner_node(observe_model(ChatCompletionsModel(settings)), settings))
        print(node(create_initial_state(args.plan, build_research_context()))["research_plan"].model_dump_json(indent=2))
        return 0
    if args.search is not None:
        print(traced_callable("web_search", "tool", build_live_search(settings))(args.search).model_dump_json(indent=2))
        return 0
    print(json.dumps({"status": "ok", "phase": 11, "langgraph_version": version("langgraph"),
        "model_name": settings.model_name, "api_key_configured": settings.api_key is not None,
        "max_retries": settings.max_retries, "writer_timeout_seconds": settings.writer_timeout_seconds,
        "embedding_provider": settings.embedding_provider, "rag_chunk_size": settings.rag_chunk_size,
        "embedding_model": settings.embedding_model,
        "rag_chunk_overlap": settings.rag_chunk_overlap, "rag_top_k": settings.rag_top_k,
        "log_level": settings.log_level, "trace_file_configured": settings.trace_file is not None}, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.eval_live and args.evaluate is None:
        parser.error("--eval-live requires --evaluate")
    if args.documents and not (args.research is not None or args.retrieve is not None or args.demo_rag):
        parser.error("--documents requires --research or --retrieve")
    if args.retrieve is not None and not args.documents:
        parser.error("--retrieve requires --documents")
    if args.local_only and not (args.research is not None and args.documents):
        parser.error("--local-only requires --research and --documents")
    if args.include_archived and not (args.documents or args.demo_rag):
        parser.error("--include-archived requires --documents")
    graph_action = args.research is not None or args.demo_graph or args.demo_critic or args.demo_rag
    if args.report_only and not graph_action:
        parser.error("--report-only requires --research")
    if args.output and not graph_action and args.evaluate is None:
        parser.error("--output requires --research or --evaluate")
    if not any((args.check_config, args.plan is not None, args.search is not None,
                args.research is not None, args.retrieve is not None, args.evaluate is not None,
                args.demo_graph, args.demo_critic, args.demo_rag, args.demo_research)):
        parser.print_help()
        return 0
    try:
        settings = load_settings()
        updates = {}
        if args.trace_file is not None:
            updates["trace_file"] = args.trace_file
        if args.log_level is not None:
            updates["log_level"] = args.log_level
        settings = settings.model_copy(update=updates)
        configure_logging(settings.log_level)
        with trace_run(settings.trace_file) as recorder, span("cli.run", "run") as attributes:
            code = execute_action(args, settings, recorder.run_id)
            attributes["exit_code"] = code
            return code
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['type']}"
                            for item in exc.errors(include_input=False, include_url=False))
        print("Configuration/dataset validation failed: " + details)
    except (RuntimeError, ValueError, OSError) as exc:
        # Unexpected exception messages can contain provider payloads or credentials.
        print(f"Command failed ({type(exc).__name__}); see trace error types and configuration.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
