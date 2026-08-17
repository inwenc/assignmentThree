"""Search API (read path) + UI + local-dev media serving.

POST /api/ask is the whole read path: retrieve -> confidence gate -> cited
multimodal answer or honest abstention (src/rag/search.py). Media endpoints
exist only for STORAGE_PROVIDER=local — with a real bucket, thumbnails and
playback stream via presigned URLs and never touch this process.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .. import config, db, llm, storage
from ..rag import search as rag_search
from .videos import require_auth, user_id as user_id_dep

router = APIRouter(tags=["search"])

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
_FRAME_RE = re.compile(r"^\d{6}\.jpg$")
_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _uid(value: str | None) -> str:
    uid = (value or config.DEFAULT_USER_ID).strip()
    if not _USER_RE.match(uid):
        raise HTTPException(400, "Invalid user id.")
    return uid


# ── Meta ─────────────────────────────────────────────────────────────────────

@router.get("/api/health")
def health():
    return {"ok": True}


@router.get("/api/config")
def get_config(x_user_id: str | None = Header(default=None)):
    cfg, source = rag_search.resolve_llm(_uid(x_user_id))
    return {
        "llm_configured": cfg is not None,
        "llm_source": source,   # "user" (their hosted model) | "server" | "none"
        "llm_provider": cfg.provider if cfg else None,
        "llm_model": cfg.model if cfg else None,
        "frame_strategy": config.FRAME_STRATEGY,
        "top_k": config.TOP_K,
        "upload_mode": "presigned" if storage.presign_capable() else "direct",
        "max_upload_mb": config.MAX_UPLOAD_MB,
    }


# ── Bring-your-own-model settings (per tenant) ────────────────────────────────
# A user points MomentSearch at THEIR hosted model — a vLLM/Ollama/LM Studio/
# Together/OpenRouter endpoint (OpenAI-compatible), NVIDIA NIM, or Anthropic —
# and every /api/ask for that user answers with it instead of the server's LLM.

class LLMSettings(BaseModel):
    provider: str = "openai"     # openai (any OpenAI-compatible) | nvidia | anthropic
    model: str                   # e.g. "Qwen/Qwen2.5-VL-7B-Instruct" on vLLM
    base_url: str | None = None  # e.g. "http://my-vllm-host:8000/v1"
    api_key: str | None = None   # empty keeps the previously stored key


def _validate_llm(s: LLMSettings) -> LLMSettings:
    if s.provider not in llm.PROVIDERS:
        raise HTTPException(400, f"provider must be one of {llm.PROVIDERS}.")
    if not s.model.strip():
        raise HTTPException(400, "model is required.")
    url = (s.base_url or "").strip()
    if url and not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(400, "base_url must be http(s).")
    if s.provider == "openai" and not url and not (s.api_key or "").strip():
        raise HTTPException(400, "Provide a base_url (your hosted endpoint) "
                                 "and/or an api_key.")
    return s


def _masked(row: dict) -> dict:
    key = row.get("api_key") or ""
    return {"provider": row["provider"], "model": row["model"],
            "base_url": row.get("base_url"),
            "api_key_set": bool(key),
            "api_key_hint": f"…{key[-4:]}" if key else None,
            "updated_at": row.get("updated_at")}


@router.get("/api/llm")
def get_llm(uid: str = Depends(user_id_dep)):
    row = db.get_user_llm(uid)
    _, source = rag_search.resolve_llm(uid)
    return {"configured": row is not None, "active_source": source,
            "settings": _masked(row) if row else None,
            "server_fallback": config.llm_configured()}


@router.put("/api/llm", dependencies=[Depends(require_auth)])
def put_llm(s: LLMSettings, uid: str = Depends(user_id_dep)):
    s = _validate_llm(s)
    row = db.set_user_llm(uid, provider=s.provider, model=s.model.strip(),
                          base_url=(s.base_url or "").strip() or None,
                          api_key=(s.api_key or "").strip())
    return {"ok": True, "settings": _masked(row)}


@router.post("/api/llm/test", dependencies=[Depends(require_auth)])
def test_llm(uid: str = Depends(user_id_dep)):
    """One tiny image through the user's model — proves connectivity AND that
    the model is vision-capable (text-only models fail here, not mid-answer)."""
    cfg, source = rag_search.resolve_llm(uid)
    if cfg is None:
        raise HTTPException(400, "No model configured.")
    try:
        reply = llm.ping(cfg)
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__}: {e}")
    return {"ok": True, "source": source, "model": cfg.model, "reply": reply[:200]}


@router.delete("/api/llm", dependencies=[Depends(require_auth)])
def delete_llm(uid: str = Depends(user_id_dep)):
    db.delete_user_llm(uid)
    return {"ok": True, "active_source": rag_search.resolve_llm(uid)[1]}


# ── Ask ──────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    video_id: str | None = None        # single-video scope (legacy)
    video_ids: list[str] | None = None  # multi-select scope (checked videos)
    top_k: int | None = None


@router.post("/api/ask")
def ask(req: AskRequest, x_user_id: str | None = Header(default=None)):
    if not req.question.strip():
        raise HTTPException(400, "Empty question.")
    # Empty list == "nothing selected" -> treat as all (None); avoids a
    # confusing zero-results answer when the user unchecks everything.
    video_ids = req.video_ids or None
    return rag_search.ask(req.question.strip(), _uid(x_user_id),
                          top_k=req.top_k, video_id=req.video_id,
                          video_ids=video_ids)


# ── Media (local-dev only; buckets serve these via presigned URLs) ───────────

@router.get("/api/frame/{video_id}/{name}")
def frame(video_id: str, name: str, u: str | None = None):
    if storage.presign_capable():
        raise HTTPException(404, "Thumbnails are served from object storage.")
    if not _FRAME_RE.match(name):
        raise HTTPException(404, "Frame not found.")
    fp = storage.local_path(f"{config.FRAME_KEY_PREFIX}{_uid(u)}/{video_id}/{name}")
    if not fp.exists():
        raise HTTPException(404, "Frame not found.")
    return FileResponse(fp, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@router.get("/api/video/{video_id}")
def video(video_id: str, u: str | None = None,
          range: str | None = Header(default=None)):
    if storage.presign_capable():
        raise HTTPException(404, "Playback streams from object storage.")
    uid = _uid(u)
    row = db.get_video(video_id)
    if row is None or row["user_id"] != uid or not row.get("storage_key"):
        raise HTTPException(404, "Video not found.")
    path = storage.local_path(row["storage_key"])
    if not path.exists():
        raise HTTPException(404, "Video file not found.")
    size = path.stat().st_size
    if range is None:
        return FileResponse(path, media_type="video/mp4",
                            headers={"Accept-Ranges": "bytes"})
    try:
        unit, rng = range.split("=", 1)
        assert unit.strip() == "bytes"
        start_s, end_s = rng.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
    except Exception:
        raise HTTPException(416, "Invalid Range header")
    if start >= size or start > end:
        raise HTTPException(416, "Range out of bounds",
                            headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)
    length = end - start + 1

    def stream():
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                buf = fh.read(min(1 << 16, remaining))
                if not buf:
                    break
                remaining -= len(buf)
                yield buf

    return StreamingResponse(stream(), status_code=206, media_type="video/mp4",
                             headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                      "Accept-Ranges": "bytes",
                                      "Content-Length": str(length)})


@router.get("/api/document/{source_id}")
def document(source_id: str, u: str | None = None):
    """Serve a stored paper/deck in local development for citation links."""
    if storage.presign_capable():
        raise HTTPException(404, "Documents are served from object storage.")
    uid = _uid(u)
    row = db.get_video(source_id)
    if (row is None or row["user_id"] != uid or row.get("source") not in {"paper", "deck"}
            or not row.get("storage_key")):
        raise HTTPException(404, "Document not found.")
    path = storage.local_path(row["storage_key"])
    if not path.exists():
        raise HTTPException(404, "Document file not found.")
    media_type = ("application/vnd.openxmlformats-officedocument.presentationml.presentation"
                  if path.suffix.lower() == ".pptx" else "application/pdf")
    return FileResponse(path, media_type=media_type,
                        headers={"Cache-Control": "private, max-age=3600"})


# ── UI ────────────────────────────────────────────────────────────────────────

def _render(mode: str) -> str:
    """Two modes of the single-page UI:
      * "sample" (/)            — curated read-only demo
      * "full"   (/get-started) — bring-your-own-videos (add URL / upload)
    """
    index = UI_DIR / "index.html"
    if not index.exists():
        return "<h1>MomentSearch</h1><p>ui/index.html not found.</p>"
    html = index.read_text(encoding="utf-8")
    return html.replace("<!--MS_MODE-->", f'<script>window.MS_MODE="{mode}";</script>')


@router.get("/", response_class=HTMLResponse)
def index():
    return _render("sample")


@router.get("/get-started", response_class=HTMLResponse)
def get_started():
    return _render("full")
