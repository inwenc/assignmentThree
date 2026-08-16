"""MomentSearch — unified API (one service, one port).

Two routers on one FastAPI app (:8000):
  - src/api/videos.py  /api/videos/*  — presigned uploads + registration +
                                        ingest status (Bearer auth)
  - src/api/search.py  public         — / (web UI), /api/ask, /api/config,
                                        local-dev media, /api/health

Heavy processing never happens here — the videos router only schedules Prefect
flow runs; worker.py (separate process, same image) executes the ingest
pipeline. Every durable byte lives in object storage, Qdrant, or Postgres, so
this process is stateless and disposable.

Run:
    uvicorn src.app:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config, db
from .api.search import router as search_router
from .api.admin import router as admin_router
from .api.videos import router as videos_router
from .rag import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    # Create the Qdrant collection up front (known CLIP dims resolve without
    # loading the model) so a question before the first ingest returns
    # "no moments" instead of a 500. Qdrant being down must not block boot.
    try:
        vector_store.ensure_collection()          # visual (CLIP frames)
        if config.ENABLE_TRANSCRIPT:
            vector_store.ensure_text_collection()  # transcript (bge text)
    except Exception as exc:
        print(f"[startup] Qdrant not ready ({exc!r}) — search degrades to empty results")
    yield


app = FastAPI(title="MomentSearch", version="1.0.0", lifespan=lifespan)
app.include_router(videos_router)
app.include_router(admin_router)
app.include_router(search_router)
