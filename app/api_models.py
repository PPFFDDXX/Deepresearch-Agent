"""Validated HTTP contracts; business state remains in app.state."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=10000)
    local_only: bool = False
    use_example_documents: bool = False
    include_archived: bool = False

    @model_validator(mode="after")
    def validate_sources(self) -> "ResearchRequest":
        if (self.local_only or self.include_archived) and not self.use_example_documents:
            raise ValueError("Local-only or archived research requires documents")
        return self


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    run_id: str | None = None
    created_at: str
    error_type: str | None = None
    result: dict | None = None
