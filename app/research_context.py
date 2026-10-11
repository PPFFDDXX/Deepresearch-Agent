"""Application-owned scope; document text cannot grant itself credibility."""
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from app.config import PROJECT_ROOT
from app.rag.models import LocalDocument


class ResearchContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: Literal["document_grounded", "real_world"] = "real_world"
    demo_source_locators: tuple[str, ...] = ()
    requirement_checks: bool = True


def build_research_context(documents: Sequence[LocalDocument] = (), *, local_only: bool = False) -> ResearchContext:
    """Recognize only repository example files, never a marker in uploaded text."""
    example_root = (PROJECT_ROOT / "examples" / "doc").resolve()
    demo = tuple(str(Path(item.locator).resolve()) for item in documents
                 if Path(item.locator).resolve().is_relative_to(example_root))
    return ResearchContext(scope="document_grounded" if local_only else "real_world",
                           demo_source_locators=demo)
