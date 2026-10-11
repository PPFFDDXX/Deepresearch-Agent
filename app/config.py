"""Validated application settings, independent of graph execution."""

from pathlib import Path

from typing import Literal

from pydantic import Field, SecretStr, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment variables override .env values and built-in defaults."""

    model_config = SettingsConfigDict(
        env_prefix="DR_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    model_name: str = Field(default="glm-4.5-flash", min_length=1)
    api_key: SecretStr | None = None
    api_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4", min_length=1)
    llm_timeout_seconds: float = Field(default=180.0, gt=0, le=300)
    planner_max_tasks: int = Field(default=5, ge=1, le=10)
    planner_max_attempts: int = Field(default=2, ge=1, le=3)
    max_retries: int = Field(default=2, ge=0, le=10)
    tool_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    search_top_k: int = Field(default=5, ge=1, le=20)
    tavily_api_key: SecretStr | None = None
    search_max_attempts: int = Field(default=2, ge=1, le=3)
    researcher_max_attempts: int = Field(default=2, ge=1, le=3)
    researcher_max_evidence: int = Field(default=10, ge=1, le=20)
    critic_max_attempts: int = Field(default=2, ge=1, le=3)
    critic_max_tasks: int = Field(default=3, ge=1, le=10)
    critic_min_score: float = Field(default=0.7, ge=0, le=1)
    writer_max_attempts: int = Field(default=2, ge=1, le=3)
    writer_timeout_seconds: float = Field(default=180.0, gt=0, le=300)
    rag_chunk_size: int = Field(default=600, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=80, ge=0, le=1000)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_min_score: float = Field(default=0.05, ge=0, le=1)
    rag_max_file_mb: int = Field(default=10, ge=1, le=100)
    rag_max_files: int = Field(default=100, ge=1, le=1000)
    embedding_provider: Literal["hashing", "api"] = "api"
    embedding_dimensions: int = Field(default=1024, ge=64, le=8192)
    embedding_model: str = Field(default="BAAI/bge-m3", min_length=1)
    embedding_base_url: str = Field(default="https://api.siliconflow.cn/v1", min_length=1)
    embedding_api_key: SecretStr | None = None
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_max_attempts: int = Field(default=2, ge=1, le=3)
    embedding_timeout_seconds: float = Field(default=60, gt=0, le=300)
    embedding_expected_dimensions: int | None = Field(default=None, ge=1, le=65536)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
    trace_file: Path | None = None

    @field_validator("api_key", "tavily_api_key", "embedding_api_key", mode="before")
    @classmethod
    def normalize_api_keys(cls, value: object) -> object:
        """Treat blank credentials as missing; never provide embedded defaults."""
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def valid_chunk_overlap(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_size")
        return self


def load_settings() -> Settings:
    """Load on demand so importing a module has no configuration side effects."""
    return Settings()
