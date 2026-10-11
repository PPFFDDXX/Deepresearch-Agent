"""FastAPI entry point: run uvicorn app.api:app --host 127.0.0.1."""
from contextlib import asynccontextmanager
from typing import Annotated, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError

from app.api_models import JobResponse, ResearchRequest
from app.config import Settings, load_settings
from app.jobs import JobCapacityError, JobManager
from app.observability import configure_logging
from app.service import execute_research


def create_app(settings: Settings | None = None, *, runner: Callable = execute_research) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        effective = settings or load_settings()
        configure_logging(effective.log_level)
        application.state.jobs = JobManager(effective, runner)
        try:
            yield
        finally:
            # Shutdown waits for submitted research; never abandon temp documents mid-run.
            from starlette.concurrency import run_in_threadpool
            await run_in_threadpool(application.state.jobs.close)

    application = FastAPI(title="DeepResearch Agent", version="0.11.0", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(error["loc"]), "type": error["type"]} for error in exc.errors()
        ]})

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "phase": 11}

    def submit(**kwargs) -> JobResponse:
        try:
            return application.state.jobs.submit(**kwargs)
        except JobCapacityError:
            raise HTTPException(429, "Research queue is full; retry later") from None

    def get_job(job_id: str) -> JobResponse:
        try:
            return application.state.jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found or expired") from None

    @application.post("/research", response_model=JobResponse, status_code=202)
    def research(body: ResearchRequest) -> JobResponse:
        return submit(query=body.query, local_only=body.local_only,
            include_archived=body.include_archived, example_documents=body.use_example_documents)

    @application.post("/research/files", response_model=JobResponse, status_code=202)
    async def research_files(query: Annotated[str, Form(min_length=1, max_length=10000)],
                             files: Annotated[list[UploadFile], File()],
                             local_only: Annotated[bool, Form()] = True,
                             include_archived: Annotated[bool, Form()] = False) -> JobResponse:
        uploads = []
        names = set()
        total = 0
        try:
            if not 1 <= len(files) <= 10:
                raise HTTPException(413, "Upload between 1 and 10 documents")
            try:
                validated = ResearchRequest(query=query, local_only=local_only,
                    include_archived=include_archived, use_example_documents=True)
            except ValidationError:
                raise HTTPException(422, "Invalid research query") from None
            for upload in files:
                name = upload.filename or ""
                if (not name or name in (".", "..") or any(c in name for c in '/\\:')
                        or any(ord(c) < 32 for c in name)
                        or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
                        or len(name) > 150 or name.endswith((".", " "))
                        or name.casefold() in names or not name.lower().endswith((".md", ".txt", ".pdf"))):
                    raise HTTPException(422, "Invalid or duplicate document filename")
                names.add(name.casefold())
                content = await upload.read(10 * 1024 * 1024 + 1)
                total += len(content)
                if not content or len(content) > 10 * 1024 * 1024 or total > 20 * 1024 * 1024:
                    raise HTTPException(413, "Documents must be nonempty, at most 10 MiB each and 20 MiB total")
                uploads.append((name, content))
            return submit(query=validated.query, local_only=local_only,
                          include_archived=include_archived, uploads=tuple(uploads))
        finally:
            for upload in files:
                await upload.close()

    @application.get("/research/{job_id}", response_model=JobResponse)
    def status(job_id: str) -> JobResponse:
        return get_job(job_id)

    @application.get("/research/{job_id}/report", response_class=PlainTextResponse)
    def report(job_id: str) -> PlainTextResponse:
        job = get_job(job_id)
        if job.status in ("queued", "running"):
            raise HTTPException(409, "Research is still running")
        if not job.result or not job.result.get("final_report"):
            raise HTTPException(409, "No report is available; inspect job status")
        return PlainTextResponse(job.result["final_report"], media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="research-report.md"'})

    return application


app = create_app()
