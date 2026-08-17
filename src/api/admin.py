"""Admin document registration and unified source status.

The document endpoints only validate and create a manifest row.  Parsing and
embedding are deliberately delegated to Prefect so a large PDF/PPTX can never
hold an API request (or a search request) open.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import config, db, jobs, storage
from .videos import require_auth, user_id

router = APIRouter(prefix="/admin", tags=["admin"])
_KINDS = {"paper", "deck"}
_DOCUMENT_PREFIX = "documents/"
_DOCUMENT_TYPES = {
    "paper": {"application/pdf"},
    "deck": {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    },
}


class DocumentRequest(BaseModel):
    uri: str = Field(min_length=1, max_length=2_048)
    kind: str
    title: str | None = Field(default=None, max_length=512)


class DocumentPresignRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=128)
    size: int = Field(gt=0)
    kind: str


def _storage_key(uri: str) -> str | None:
    """Convert the documented ``storage://path/to/file`` form to an object key."""
    if not uri.startswith("storage://"):
        return None
    key = uri.removeprefix("storage://").lstrip("/")
    if not key or ".." in Path(key).parts:
        raise HTTPException(400, "Invalid storage URI.")
    return key


def _validate_uri(uri: str, kind: str) -> tuple[str | None, str | None]:
    storage_key = _storage_key(uri)
    if storage_key is not None:
        target = storage_key
    else:
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(400, "uri must be an http(s) URL or storage:// object reference.")
        target = parsed.path
    suffix = Path(target).suffix.lower()
    if kind == "paper" and suffix != ".pdf":
        raise HTTPException(400, "Papers must be PDF files.")
    if kind == "deck" and suffix not in {".pdf", ".pptx"}:
        raise HTTPException(400, "Decks must be PDF or PPTX files.")
    return storage_key, None if storage_key is not None else uri


@router.post("/documents/presign", dependencies=[Depends(require_auth)])
def presign_document(req: DocumentPresignRequest, uid: str = Depends(user_id)):
    """Return a direct-upload target for a PDF paper or PDF/PPTX deck."""
    kind = req.kind.strip().lower()
    if kind not in _KINDS:
        raise HTTPException(400, "kind must be 'paper' or 'deck'.")
    if req.size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_MB}MB limit.")
    ext = Path(req.filename).suffix.lower()
    permitted = {".pdf"} if kind == "paper" else {".pdf", ".pptx"}
    if ext not in permitted:
        raise HTTPException(415, f"{kind.capitalize()} uploads must be {', '.join(sorted(permitted))} files.")
    # Browsers sometimes provide an empty/generic type for .pptx, so extension
    # validation is authoritative while known MIME types get an extra check.
    if req.content_type not in _DOCUMENT_TYPES[kind] and req.content_type not in {
        "", "application/octet-stream"
    }:
        raise HTTPException(415, "Unsupported document content type.")
    upload_id = f"docup_{uuid.uuid4().hex[:12]}"
    key = f"{_DOCUMENT_PREFIX}{uid}/{upload_id}{ext}"
    content_type = req.content_type if req.content_type in _DOCUMENT_TYPES[kind] else "application/octet-stream"
    if storage.presign_capable():
        return {"mode": "presigned", "upload_id": upload_id, "key": key,
                **storage.presign_put(key, content_type)}
    return {"mode": "direct", "upload_id": upload_id, "key": key,
            "url": f"/admin/documents/{upload_id}/content?key={key}",
            "headers": {"Content-Type": content_type}}


@router.put("/documents/{upload_id}/content", dependencies=[Depends(require_auth)])
async def upload_document_direct(upload_id: str, key: str, request: Request,
                                 uid: str = Depends(user_id)):
    """Local-storage fallback for the browser's document upload."""
    if storage.presign_capable():
        raise HTTPException(400, "Use the presigned URL to upload.")
    if not key.startswith(f"{_DOCUMENT_PREFIX}{uid}/{upload_id}"):
        raise HTTPException(403, "Key does not belong to this document upload.")
    destination = storage.local_path(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_MB}MB limit.")
            out.write(chunk)
    return {"ok": True, "key": key, "size": size}


@router.post("/documents", status_code=202, dependencies=[Depends(require_auth)])
def register_document(req: DocumentRequest, uid: str = Depends(user_id)):
    kind = req.kind.strip().lower()
    if kind not in _KINDS:
        raise HTTPException(400, "kind must be 'paper' or 'deck'.")
    uri = req.uri.strip()
    storage_key, url = _validate_uri(uri, kind)
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    title = req.title.strip() if req.title and req.title.strip() else Path(
        storage_key or urlparse(url or "").path).stem
    row = db.upsert_pending({"id": document_id, "user_id": uid, "source": kind,
                             "url": url, "storage_key": storage_key,
                             "source_hash": None, "title": title})

    # Match video registration: fair mode leaves a row pending for the worker's
    # dispatcher; FIFO mode schedules the named document deployment immediately.
    if config.ENABLE_FAIR_DISPATCH:
        return {"id": row["id"], "status": "pending", "kind": kind}
    try:
        flow_run_id = jobs.enqueue_document(row["id"], uid, kind)
    except Exception as exc:
        # The row remains pending and can be retried; the caller receives a
        # useful upstream-failure status instead of a misleading accepted run.
        raise HTTPException(502, f"Could not schedule document ingestion: {exc}") from exc
    return {"id": row["id"], "status": "pending", "kind": kind,
            "flow_run_id": flow_run_id}


@router.get("/sources", dependencies=[Depends(require_auth)])
def list_sources(uid: str = Depends(user_id)):
    """One status list for legacy videos and new paper/deck source rows."""
    sources = []
    for row in db.list_videos(uid):
        source = row.get("source")
        kind = source if source in _KINDS else "video"
        sources.append({"id": row["id"], "kind": kind, "status": row.get("status"),
                        "title": row.get("title"), "pct": row.get("progress"),
                        "error": row.get("error"), "created_at": row.get("created_at"),
                        "updated_at": row.get("updated_at")})
    return {"sources": sources}
