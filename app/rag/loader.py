"""Load UTF-8 Markdown/TXT and text PDFs; preserve file and PDF page identity."""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from app.config import Settings
from app.rag.models import DocumentPage, LocalDocument, RagError


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def discover_files(paths: Sequence[Path], max_files: int) -> list[Path]:
    """Expand directories deterministically and deduplicate resolved file paths."""
    found: set[Path] = set()
    for supplied in paths:
        path = supplied.expanduser().resolve()
        if not path.exists():
            raise RagError(f"Document path does not exist: {path}")
        if path.is_dir():
            candidates = path.rglob("*")
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            candidates = [path]
        else:
            raise RagError(f"Unsupported document input: {path.name}")
        for item in candidates:
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES:
                resolved = item.resolve()
                if path.is_dir() and not resolved.is_relative_to(path):
                    raise RagError("Document directory contains a link outside its root")
                found.add(resolved)
                if len(found) > max_files:
                    raise RagError(f"Document count exceeds limit ({max_files})")
    if not found:
        raise RagError("No supported Markdown, TXT or PDF documents found")
    return sorted(found, key=lambda item: str(item).casefold())


def extract_metadata(text: str, path: Path) -> tuple[str, bool, str | None, str | None]:
    """Read demonstrator header fields; this is not an authorization/version engine."""
    first_line = next((line.strip().lstrip("# ") for line in text.splitlines() if line.strip()), path.stem)
    header = text.split("\n## ", 1)[0][:1500]
    def field(name: str) -> str | None:
        match = re.search(rf"(?m)^\s*[-*]?\s*{name}\s*[:：]\s*(.+)$", header)
        return match.group(1).strip() if match else None
    status = field("状态") or ""
    archived = any(word in status for word in ("归档", "失效", "废止")) or "历史版本" in first_line
    return first_line[:200], archived, field("版本"), field("生效日期")


def load_document(path: Path, settings: Settings) -> LocalDocument:
    """Reject empty/oversized/unreadable inputs instead of indexing guessed text."""
    try:
        if path.stat().st_size > settings.rag_max_file_mb * 1024 * 1024:
            raise RagError(f"Document exceeds size limit: {path.name}")
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise RagError(f"Encrypted PDF is not supported: {path.name}")
            pages = []
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(DocumentPage(text=text, page_number=number))
        else:
            text = path.read_text(encoding="utf-8-sig").strip()
            pages = [DocumentPage(text=text)] if text else []
    except RagError:
        raise
    except Exception as exc:
        # Do not include parser internals or document contents in error output.
        raise RagError(f"Could not read {path.name} ({type(exc).__name__})") from None
    if not pages:
        raise RagError(f"No extractable text in {path.name}; scanned PDFs require external OCR")
    title, archived, version, effective_date = extract_metadata(pages[0].text, path)
    locator = str(path.resolve())
    return LocalDocument(id="doc-" + hashlib.sha256(locator.encode()).hexdigest(), title=title,
                         locator=locator, archived=archived, version=version,
                         effective_date=effective_date, pages=pages)


def load_documents(paths: Sequence[Path], settings: Settings) -> list[LocalDocument]:
    """Fail the ingestion batch if an explicitly selected document cannot be read."""
    return [load_document(path, settings) for path in discover_files(paths, settings.rag_max_files)]
