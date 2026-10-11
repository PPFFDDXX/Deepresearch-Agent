"""Bounded in-process job execution; no persistent queue or cross-worker sharing."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Callable
from uuid import uuid4

from app.api_models import JobResponse
from app.config import PROJECT_ROOT, Settings
from app.observability import span, trace_run
from app.service import execute_research


class JobCapacityError(RuntimeError):
    pass


class JobManager:
    def __init__(self, settings: Settings, runner: Callable = execute_research,
                 *, workers: int = 2, max_pending: int = 8, max_records: int = 100):
        self.settings = settings
        self.runner = runner
        self.max_pending = max_pending
        self.max_records = max_records
        self.jobs: dict[str, JobResponse] = {}
        self.lock = Lock()
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="research")

    def submit(self, query: str, *, local_only: bool = False, include_archived: bool = False,
               example_documents: bool = False, uploads: tuple[tuple[str, bytes], ...] = ()) -> JobResponse:
        with self.lock:
            active = sum(job.status in ("queued", "running") for job in self.jobs.values())
            if self.closed or active >= self.max_pending:
                raise JobCapacityError("Research queue is full or shutting down")
            if len(self.jobs) >= self.max_records:
                terminal = next((key for key, job in self.jobs.items() if job.status in ("completed", "failed")), None)
                if terminal is None:
                    raise JobCapacityError("Research storage is full")
                del self.jobs[terminal]
            job = JobResponse(job_id=str(uuid4()), status="queued", created_at=datetime.now(timezone.utc).isoformat())
            self.jobs[job.job_id] = job
            try:
                self.executor.submit(self._execute, job.job_id, query, local_only, include_archived, example_documents, uploads)
            except RuntimeError:
                del self.jobs[job.job_id]
                raise JobCapacityError("Research executor is unavailable") from None
            return job.model_copy(deep=True)

    def get(self, job_id: str) -> JobResponse:
        with self.lock:
            return self.jobs[job_id].model_copy(deep=True)

    def _update(self, job_id: str, **changes) -> None:
        with self.lock:
            self.jobs[job_id] = self.jobs[job_id].model_copy(update=deepcopy(changes))

    def _execute(self, job_id: str, query: str, local_only: bool, include_archived: bool,
                 example_documents: bool, uploads: tuple[tuple[str, bytes], ...]) -> None:
        with trace_run(self.settings.trace_file) as recorder:
            self._update(job_id, status="running", run_id=recorder.run_id)
            try:
                with span("api.research", "run") as attributes, TemporaryDirectory(prefix="deepresearch-") as folder:
                    documents = [PROJECT_ROOT / "examples" / "doc"] if example_documents else []
                    for name, content in uploads:
                        path = Path(folder) / name
                        path.write_bytes(content)
                        documents.append(path)
                    result = self.runner(query, self.settings, documents=documents,
                        local_only=local_only, include_archived=include_archived)
                    failed = result.get("research_status") == "critic_error" or result.get("report_status") == "writer_error"
                    attributes["exit_code"] = int(failed)
                    self._update(job_id, status="failed" if failed else "completed", result=result,
                                 error_type="ResearchStageError" if failed else None)
            except Exception as exc:
                self._update(job_id, status="failed", error_type=type(exc).__name__)

    def close(self) -> None:
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
