"""Asynchronous, page-aware PDF paper ingestion.

This mirrors :mod:`src.ingest.pipeline`'s video flow, but a paper's durable
locator is a one-based PDF page number rather than a video timestamp.  Chunks
never cross a page boundary, so the ``page`` stored in Qdrant is always a real
citation target.

The manifest row is intentionally read through the existing ``ms_videos``
access layer.  Document API wiring can store a paper's object key in
``storage_key`` or its public PDF URL in ``url``; using the shared ``id`` field
also lets the existing vector-store delete/upsert operations remain idempotent.
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from prefect import flow, task

from .. import db, storage
from ..config import TEXT_EMBED_VERSION
from ..rag import vector_store
from ..rag.embeddings import embed_docs
from .fetch import scratch_dir, sha256_file


@dataclass(frozen=True)
class PaperPage:
    """Extracted text from a PDF page; ``number`` is one-based for citations."""

    number: int
    text: str


_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+|[A-Z][A-Z0-9 ,:;\-]{4,})")


def extract_pages(path: Path) -> list[PaperPage]:
    """Extract selectable PDF text while retaining its original page number.

    PyMuPDF is used rather than flattening the entire PDF: it is fast, handles
    common academic PDFs well, and exposes pages directly.  A scanned PDF with
    no text produces no chunks and fails clearly instead of inventing OCR text.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # helpful when a worker image missed requirements
        raise RuntimeError("PDF ingestion requires PyMuPDF (pip install pymupdf)") from exc

    pages: list[PaperPage] = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf):
            # ``sort=True`` makes multi-column extraction substantially more
            # readable while preserving page-local citation semantics.
            text = "\n".join(line.strip() for line in page.get_text("text", sort=True).splitlines()
                             if line.strip())
            if text:
                pages.append(PaperPage(number=index + 1, text=text))
    return pages


def chunk_pages(pages: list[PaperPage], *, max_chars: int = 1_600,
                overlap_chars: int = 180) -> list[dict]:
    """Make paragraph/heading-aware chunks without crossing PDF pages.

    Character limits deliberately keep this independent of the configured
    embedding provider's tokenizer.  Paragraph boundaries and apparent section
    headings are preferred split points; long paragraphs fall back to word
    boundaries.  Every result includes a one-based ``page`` locator.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    chunks: list[dict] = []
    for page in pages:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", page.text) if b.strip()]
        # PyMuPDF text extraction commonly emits one line per visual line; when
        # it did not preserve blank paragraphs, build sensible blocks from lines.
        if len(blocks) == 1:
            blocks = [line.strip() for line in page.text.splitlines() if line.strip()]

        current = ""
        for block in blocks:
            # Preserve a heading with the following prose where possible.
            separator = "\n\n" if current else ""
            candidate = current + separator + block
            if current and (len(candidate) > max_chars or _HEADING_RE.match(block)):
                chunks.extend(_split_text(current, page.number, max_chars, overlap_chars))
                current = block
            else:
                current = candidate
        if current:
            chunks.extend(_split_text(current, page.number, max_chars, overlap_chars))
    return chunks


def _split_text(text: str, page: int, max_chars: int, overlap_chars: int) -> list[dict]:
    """Split a page-local block on whitespace, adding a bounded text overlap."""
    text = " ".join(text.split())
    results: list[dict] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            results.append({"page": page, "text": piece})
        if end >= len(text):
            break
        # A non-whitespace-free page always advances; the defensive max covers
        # pathological input such as a single enormous identifier.
        start = max(end - overlap_chars, start + 1)
    return results


def fetch_paper(storage_key: str | None, url: str | None, paper_id: str) -> Path:
    """Acquire a PDF into worker scratch space from object storage or HTTPS."""
    if storage_key:
        return storage.download_to(storage_key, scratch_dir() / f"{paper_id}.pdf")
    if not url or not url.lower().startswith(("https://", "http://")):
        raise ValueError("paper requires a storage_key or an http(s) PDF URL")
    dest = scratch_dir() / f"{paper_id}.pdf"
    request = urllib.request.Request(url, headers={"User-Agent": "MomentSearch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as out:
            while data := response.read(1 << 20):
                out.write(data)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest


@task(name="fetch-paper", retries=2, retry_delay_seconds=[30, 120])
def t_fetch_paper(paper_id: str, user_id: str) -> str:
    row = db.get_video(paper_id)
    if row is None:
        raise ValueError(f"no manifest row for {paper_id}")
    db.set_status(paper_id, "parsing", progress=0.0)
    path = fetch_paper(row.get("storage_key"), row.get("url"), paper_id)
    source_hash = sha256_file(path)
    db.set_status(paper_id, "parsing", source_hash=source_hash, progress=0.05)
    duplicate = db.find_duplicate(user_id, source_hash, exclude_id=paper_id)
    if duplicate:
        path.unlink(missing_ok=True)
        db.set_status(paper_id, "skipped", error=f"duplicate of {duplicate['id']}")
        return ""
    return str(path)


@task(name="parse-paper", retries=1, retry_delay_seconds=30)
def t_parse_paper(paper_id: str, path: str) -> list[PaperPage]:
    db.set_status(paper_id, "parsing", progress=0.1)
    pages = extract_pages(Path(path))
    if not pages:
        raise RuntimeError("No selectable text could be extracted from this PDF.")
    db.set_progress(paper_id, 1.0)
    return pages


@task(name="chunk-paper")
def t_chunk_paper(paper_id: str, pages: list[PaperPage]) -> list[dict]:
    db.set_status(paper_id, "chunking", progress=0.0)
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError("PDF text extraction produced no indexable chunks.")
    db.set_progress(paper_id, 1.0)
    return chunks


@task(name="enrich-paper")
def t_enrich_paper(paper_id: str, chunks: list[dict]) -> list[dict]:
    """Status seam for the shared enrichment stage.

    This codebase has no document-enrichment implementation yet.  Keeping this
    identity task preserves the paper lifecycle and is the single place to call
    it when one is added, without changing parsing or vector writes.
    """
    db.set_status(paper_id, "enriching", progress=1.0)
    return chunks


@task(name="embed-index-paper", retries=2, retry_delay_seconds=60)
def t_embed_index_paper(paper_id: str, user_id: str, chunks: list[dict]) -> int:
    db.set_status(paper_id, "embedding", progress=0.0)
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, paper_id)  # re-runs replace, never duplicate

    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embed_docs([chunk["text"] for chunk in batch])
        vector_store.upsert_chunks(user_id, paper_id, vectors, payloads=[
            {"user_id": user_id, "video_id": paper_id, "source_id": paper_id,
             "kind": "paper", "modality": "text", "page": chunk["page"],
             "text": chunk["text"], "embed_version": TEXT_EMBED_VERSION}
            for chunk in batch
        ])
        db.set_progress(paper_id, min(1.0, (start + len(batch)) / len(chunks)))
    db.set_status(paper_id, "indexed", frame_count=len(chunks),
                  embed_version=TEXT_EMBED_VERSION, progress=1.0)
    return len(chunks)


@flow(name="ms-ingest-paper", log_prints=True, timeout_seconds=3600)
def ingest_paper(paper_id: str, user_id: str) -> dict:
    """Prefect flow for an already-registered paper manifest row."""
    attempt = db.bump_attempts(paper_id)
    path: str | None = None
    try:
        path = t_fetch_paper(paper_id, user_id)
        if not path:
            return {"paper_id": paper_id, "skipped": True}
        pages = t_parse_paper(paper_id, path)
        chunks = t_chunk_paper(paper_id, pages)
        chunks = t_enrich_paper(paper_id, chunks)
        count = t_embed_index_paper(paper_id, user_id, chunks)
        print(f"[paper] {paper_id} indexed: {count} chunks ({len(pages)} pages, attempt {attempt})")
        return {"paper_id": paper_id, "pages": len(pages), "chunks": count}
    except Exception as exc:
        db.set_status(paper_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if path:
            Path(path).unlink(missing_ok=True)
