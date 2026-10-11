"""Section-aware character windows; excerpts always remain original substrings."""

import hashlib
import re

from app.rag.models import DocumentChunk, LocalDocument


def split_sections(text: str) -> list[tuple[str, str]]:
    """Retain headings; recognize Markdown and demonstrator TXT section labels."""
    matches = list(re.finditer(r"(?m)^(?:#{1,6}\s+[^\n]+|第\s*\d+\s*节[^\n]*)$", text))
    if not matches:
        return [("document", text)]
    sections = []
    if text[:matches[0].start()].strip():
        sections.append(("preamble", text[:matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group().lstrip("# ").strip(), text[match.start():end]))
    return sections


def chunk_document(document: LocalDocument, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    """Never cross PDF pages/sections; overlap only windows within one section."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks = []
    for page in document.pages:
        for heading, section in split_sections(page.text):
            start = 0
            while start < len(section):
                end = min(start + chunk_size, len(section))
                text = section[start:end].strip()
                if text:
                    identity = f"{document.id}|{page.page_number}|{heading}|{start}|{text}"
                    chunk_id = "chunk-" + hashlib.sha256(identity.encode()).hexdigest()
                    location = (f"page {page.page_number}; " if page.page_number else "")
                    location += f"section {heading}; chars {start + 1}-{end}; {chunk_id}"
                    if document.version:
                        location += f"; version {document.version}"
                    if document.effective_date:
                        location += f"; effective {document.effective_date}"
                    chunks.append(DocumentChunk(id=chunk_id, document_id=document.id, title=document.title,
                                                locator=document.locator, text=text, location=location,
                                                archived=document.archived))
                if end == len(section):
                    break
                start = end - overlap
    return chunks
