"""Local metadata-only tracing with ContextVar isolation and standard logging."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from app.llm import StructuredModel


LOGGER = logging.getLogger("deepresearch")
_RECORDER: ContextVar[TraceRecorder | None] = ContextVar("research_recorder", default=None)
_PARENT: ContextVar[str | None] = ContextVar("research_parent_span", default=None)
P = ParamSpec("P")
R = TypeVar("R")
_COUNTS = {"attempt", "round", "retry_count", "task_count", "evidence_count", "citation_count",
           "outcome_count", "result_count", "input_count", "output_count", "input_characters",
           "output_characters", "dimensions", "exit_code", "http_status", "case_count", "document_count", "chunk_count"}
_STATUSES = {"research_status": {"running", "sufficient", "insufficient", "critic_error"},
             "report_status": {"not_started", "generated", "partial", "no_evidence", "writer_error"}}


def configure_logging(level: str = "INFO") -> None:
    """Configure only our logger; reserve stdout for reports and JSON."""
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    if not any(getattr(handler, "_research_handler", False) for handler in LOGGER.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._research_handler = True
        LOGGER.addHandler(handler)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def safe_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Allowlist counts/statuses, never arbitrary strings or nested payloads."""
    output = {}
    for key, value in attributes.items():
        if key in _COUNTS and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            output[key] = value
        elif key in _STATUSES and isinstance(value, str) and value in _STATUSES[key]:
            output[key] = value
    return output


@dataclass
class TraceRecorder:
    trace_file: Path | None = None
    run_id: str = field(default_factory=lambda: str(uuid4()))
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, event: dict[str, Any]) -> None:
        """Serialize trusted span envelope plus filtered metadata, append one JSONL row."""
        fields = {"span_id", "parent_span_id", "name", "kind", "started_at", "duration_ms", "status", "error_type"}
        clean = {**{key: value for key, value in event.items() if key in fields}, "run_id": self.run_id,
                 "attributes": safe_attributes(event.get("attributes", {}))}
        text = json.dumps(clean, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self.events.append(clean)
            if self.trace_file is not None:
                try:
                    self.trace_file.parent.mkdir(parents=True, exist_ok=True)
                    with self.trace_file.open("a", encoding="utf-8") as stream:
                        stream.write(text + "\n")
                except OSError as exc:
                    # Observability must not mask the business exception or destroy a report.
                    LOGGER.warning("trace_write_failed error_type=%s", type(exc).__name__)
        LOGGER.info(text)


def current_recorder() -> TraceRecorder | None:
    return _RECORDER.get()


@contextmanager
def trace_run(trace_file: Path | None = None) -> Iterator[TraceRecorder]:
    """Start an isolated run; restore previous context even after exceptions."""
    recorder = TraceRecorder(trace_file=trace_file)
    recorder_token = _RECORDER.set(recorder)
    parent_token = _PARENT.set(None)
    try:
        yield recorder
    finally:
        _PARENT.reset(parent_token)
        _RECORDER.reset(recorder_token)


@contextmanager
def span(name: str, kind: str) -> Iterator[dict[str, Any]]:
    """Record one completed operation; exception type only, no message/stack/inputs."""
    attributes: dict[str, Any] = {}
    recorder = current_recorder()
    if recorder is None:
        yield attributes
        return
    safe_name = name if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", name) else "operation"
    safe_kind = kind if kind in {"run", "node", "tool", "model", "embedding", "rag", "evaluation"} else "operation"
    span_id = str(uuid4())
    parent_id = _PARENT.get()
    parent_token = _PARENT.set(span_id)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    error_type = None
    try:
        yield attributes
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        _PARENT.reset(parent_token)
        recorder.record({"span_id": span_id, "parent_span_id": parent_id, "name": safe_name,
                         "kind": safe_kind, "started_at": started_at,
                         "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                         "status": "error" if error_type or attributes.get("exit_code", 0) else "ok", "error_type": error_type,
                         "attributes": attributes})


def result_attributes(result: Any) -> dict[str, Any]:
    """Summarize node deltas/tool outputs, never stringify the returned data."""
    if isinstance(result, (list, tuple)):
        return {"result_count": len(result)}
    if isinstance(result, dict):
        output = {}
        for key, count_key in (("evidence", "evidence_count"), ("citations", "citation_count"),
                               ("research_outcomes", "outcome_count")):
            if isinstance(result.get(key), list):
                output[count_key] = len(result[key])
        for key in ("research_status", "report_status", "retry_count"):
            if key in result:
                output[key] = result[key]
        plan = result.get("research_plan")
        if plan is not None and hasattr(plan, "tasks"):
            output["task_count"] = len(plan.tasks)
        if output.get("research_status") == "critic_error" or output.get("report_status") == "writer_error":
            output["exit_code"] = 1
        return safe_attributes(output)
    if hasattr(result, "results") and isinstance(result.results, list):
        return {"result_count": len(result.results)}
    return {}


def traced_callable(name: str, kind: str, function: Callable[P, R]) -> Callable[P, R]:
    """Wrap a node/tool without modifying its input, output or exception behavior."""
    @wraps(function)
    def traced(*args: P.args, **kwargs: P.kwargs) -> R:
        with span(name, kind) as attributes:
            result = function(*args, **kwargs)
            attributes.update(result_attributes(result))
            return result
    return traced


class ObservedModel:
    def __init__(self, model: StructuredModel):
        self.model = model

    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        with span("llm.generate", "model") as attributes:
            attributes["input_characters"] = len(system) + len(user)
            output = self.model.generate(system=system, user=user, schema=schema)
            attributes["output_characters"] = len(output)
            return output


def observe_model(model: StructuredModel) -> StructuredModel:
    """Idempotent wrapper; shared Researcher/Writer models are not double traced."""
    return model if isinstance(model, ObservedModel) else ObservedModel(model)
