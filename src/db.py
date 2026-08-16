"""Postgres (Neon) access layer — the videos manifest, source of truth.

One row per (user's) video; `status` tracks the ingest lifecycle:
pending -> fetching -> sampling -> embedding -> indexed | skipped | failed
(skipped = duplicate (user_id, source_hash); indexed = searchable in Qdrant).
"""
from __future__ import annotations

import os
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL, INFLIGHT_STATUSES

_pool: ConnectionPool | None = None
_pool_pid: int | None = None


def pool() -> ConnectionPool:
    """Process-local pool. Prefect runs flows in subprocesses; a child must
    never reuse the parent's SSL connections (corrupts the TLS stream), so a
    fork gets a fresh pool."""
    global _pool, _pool_pid
    if _pool is None or _pool_pid != os.getpid():
        # check= pings each connection before lending it out — Neon silently
        # drops idle SSL connections, which otherwise 500s the first request
        # after a quiet period.
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5,
                               check=ConnectionPool.check_connection,
                               kwargs={"row_factory": dict_row})
        _pool_pid = os.getpid()
    return _pool


SCHEMA = """
CREATE TABLE IF NOT EXISTS ms_videos (
    id           TEXT PRIMARY KEY,           -- yt_<id> | up_<uuid>
    user_id      TEXT NOT NULL,
    source       TEXT NOT NULL,              -- youtube | upload
    url          TEXT,                       -- YouTube URL (source=youtube)
    storage_key  TEXT,                       -- uploads/<user>/<id>.<ext> (source=upload)
    source_hash  TEXT,                       -- sha256 of the file / yt video id
    title        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    frame_count  INT,
    progress     REAL,                       -- 0..1 within the current stage
    attempts     INT NOT NULL DEFAULT 0,
    embed_version TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ms_videos_user_idx   ON ms_videos (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ms_videos_status_idx ON ms_videos (status);
CREATE INDEX IF NOT EXISTS ms_videos_hash_idx   ON ms_videos (user_id, source_hash);

-- Bring-your-own-model: a tenant's hosted LLM endpoint (vLLM / Ollama / any
-- OpenAI-compatible server, NVIDIA NIM, or Anthropic). When a row exists the
-- read path answers with THIS model instead of the server's LLM_* env config.
CREATE TABLE IF NOT EXISTS ms_user_llms (
    user_id    TEXT PRIMARY KEY,
    provider   TEXT NOT NULL DEFAULT 'openai',  -- openai | nvidia | anthropic
    model      TEXT NOT NULL,
    base_url   TEXT,                            -- e.g. http://my-vllm:8000/v1
    api_key    TEXT,                            -- optional (vLLM often has none)
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_schema() -> None:
    with pool().connection() as conn:
        conn.execute(SCHEMA)


def upsert_pending(video: dict[str, Any]) -> dict:
    """Insert a video as pending; re-submitting an existing id resets it."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO ms_videos (id, user_id, source, url, storage_key, source_hash, title, status)
            VALUES (%(id)s, %(user_id)s, %(source)s, %(url)s, %(storage_key)s,
                    %(source_hash)s, %(title)s, 'pending')
            ON CONFLICT (id) DO UPDATE SET
                url = COALESCE(EXCLUDED.url, ms_videos.url),
                storage_key = COALESCE(EXCLUDED.storage_key, ms_videos.storage_key),
                source_hash = COALESCE(EXCLUDED.source_hash, ms_videos.source_hash),
                title = COALESCE(EXCLUDED.title, ms_videos.title),
                status = 'pending', error = NULL, progress = NULL, updated_at = now()
            RETURNING *
            """,
            video,
        ).fetchone()
    return row


def set_status(video_id: str, status: str, *, error: str | None = None,
               title: str | None = None, frame_count: int | None = None,
               source_hash: str | None = None, embed_version: str | None = None,
               progress: float | None = None) -> None:
    with pool().connection() as conn:
        conn.execute(
            """
            UPDATE ms_videos SET status = %s, error = %s,
                title = COALESCE(%s, title),
                frame_count = COALESCE(%s, frame_count),
                source_hash = COALESCE(%s, source_hash),
                embed_version = COALESCE(%s, embed_version),
                progress = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (status, error, title, frame_count, source_hash, embed_version,
             progress, video_id),
        )


def set_progress(video_id: str, progress: float) -> None:
    with pool().connection() as conn:
        conn.execute("UPDATE ms_videos SET progress = %s, updated_at = now() WHERE id = %s",
                     (round(progress, 3), video_id))


def bump_attempts(video_id: str) -> int:
    with pool().connection() as conn:
        row = conn.execute(
            "UPDATE ms_videos SET attempts = attempts + 1, updated_at = now() WHERE id = %s RETURNING attempts",
            (video_id,),
        ).fetchone()
    return row["attempts"] if row else 0


def get_video(video_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM ms_videos WHERE id = %s", (video_id,)).fetchone()


def find_duplicate(user_id: str, source_hash: str, exclude_id: str) -> dict | None:
    """An already-indexed video with the same content for the same user."""
    with pool().connection() as conn:
        return conn.execute(
            """
            SELECT * FROM ms_videos
            WHERE user_id = %s AND source_hash = %s AND id <> %s AND status = 'indexed'
            LIMIT 1
            """,
            (user_id, source_hash, exclude_id),
        ).fetchone()


def list_videos(user_id: str, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM ms_videos WHERE user_id = %s"
    params: list = [user_id]
    if status:
        q += " AND status = %s"
        params.append(status)
    q += " ORDER BY created_at DESC"
    with pool().connection() as conn:
        return conn.execute(q, tuple(params)).fetchall()


def videos_by_ids(ids: list[str]) -> dict[str, dict]:
    """Metadata join for search citations (title/url/source live here, not in Qdrant)."""
    if not ids:
        return {}
    with pool().connection() as conn:
        rows = conn.execute("SELECT * FROM ms_videos WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: r for r in rows}


def delete_video(video_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("DELETE FROM ms_videos WHERE id = %s", (video_id,))


# ── Fair scheduling (WFQ) ────────────────────────────────────────────────────

def count_inflight() -> int:
    """How many videos currently occupy execution capacity (scheduled/running)."""
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM ms_videos WHERE status = ANY(%s)",
            (list(INFLIGHT_STATUSES),),
        ).fetchone()
    return row["n"] if row else 0


def wfq_claim(limit: int) -> list[dict]:
    """Atomically claim up to `limit` pending videos in FAIR (round-robin across
    users) order, flipping them pending -> queued. Returns the claimed rows.

    Fairness: rank each user's pending videos by age (row_number partitioned by
    user_id), then order by that rank first — so we take everyone's oldest, then
    everyone's 2nd, ... A user who dumped 50 videos only gets one slot per round,
    exactly like the others. The UPDATE ... WHERE status='pending' RETURNING is
    the atomic claim: if two dispatchers race, each row is handed out once.
    """
    if limit <= 0:
        return []
    with pool().connection() as conn:
        picked = conn.execute(
            """
            SELECT id FROM (
                SELECT id, row_number() OVER (
                    PARTITION BY user_id ORDER BY created_at, id) AS rn
                FROM ms_videos WHERE status = 'pending'
            ) t
            ORDER BY rn, id
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        ids = [r["id"] for r in picked]
        if not ids:
            return []
        return conn.execute(
            """
            UPDATE ms_videos SET status = 'queued', updated_at = now()
            WHERE id = ANY(%s) AND status = 'pending'
            RETURNING id, user_id, source
            """,
            (ids,),
        ).fetchall()


# ── Bring-your-own-model (per-tenant LLM endpoint) ───────────────────────────

def get_user_llm(user_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM ms_user_llms WHERE user_id = %s",
                            (user_id,)).fetchone()


def set_user_llm(user_id: str, *, provider: str, model: str,
                 base_url: str | None, api_key: str | None) -> dict:
    """Upsert a tenant's model endpoint. An empty api_key keeps the stored one
    (so users can change model/URL without re-pasting their secret)."""
    with pool().connection() as conn:
        return conn.execute(
            """
            INSERT INTO ms_user_llms (user_id, provider, model, base_url, api_key)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                base_url = EXCLUDED.base_url,
                api_key = COALESCE(NULLIF(EXCLUDED.api_key, ''), ms_user_llms.api_key),
                updated_at = now()
            RETURNING *
            """,
            (user_id, provider, model, base_url, api_key),
        ).fetchone()


def delete_user_llm(user_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("DELETE FROM ms_user_llms WHERE user_id = %s", (user_id,))
