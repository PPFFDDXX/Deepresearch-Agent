"""Provider-independent search data; no AgentState or task IDs here."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WebSearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    title: str = Field(min_length=1)
    url: HttpUrl
    snippet: str = Field(min_length=1)


class WebSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    provider: str
    results: list[WebSearchHit]
    attempts: int = Field(ge=1)


class ResearchMaterial(BaseModel):
    """Normalized input to evidence extraction, including future local retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    title: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    source_type: Literal["web", "local"]
    text: str = Field(min_length=1)
    location: str | None = None
    origin: Literal["search_snippet", "local_chunk", "webpage"]


class ToolError(RuntimeError):
    """Safe error details for callers; no raw HTTP body or credential included."""

    def __init__(
        self, code: Literal["configuration", "invalid_input", "timeout", "network", "http", "invalid_response"],
        message: str, *, retryable: bool = False, attempts: int = 0,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.attempts = attempts
