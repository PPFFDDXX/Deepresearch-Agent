"""Validated local document units, chunks and nearest-neighbor results."""

from pydantic import BaseModel, ConfigDict, Field


class RagRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DocumentPage(RagRecord):
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class LocalDocument(RagRecord):
    id: str
    title: str
    locator: str
    archived: bool
    version: str | None
    effective_date: str | None
    pages: list[DocumentPage] = Field(min_length=1)


class DocumentChunk(RagRecord):
    id: str
    document_id: str
    title: str
    locator: str
    text: str = Field(min_length=1)
    location: str
    archived: bool


class ChunkHit(RagRecord):
    chunk: DocumentChunk
    score: float


class RagError(RuntimeError):
    """Safe ingestion, embedding or index error exposed at the application boundary."""
