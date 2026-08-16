"""Prefect Cloud trigger layer — the API schedules runs, workers execute them.

One flow ("ms-ingest-video" — the "ms-" prefix keeps it distinct from the
digital-twin-akash flow living in the same Prefect workspace), one deployment
("ingest", registered by worker.py's flow.serve()). The API never imports the
pipeline or its heavy deps (torch, ffmpeg) — it just asks Prefect Cloud to
schedule a run; any live worker picks it up. Retries/backoff live on the
flow's tasks (src/ingest/pipeline.py); failed runs are visible + retryable in
the Prefect Cloud UI.
"""
from __future__ import annotations

from prefect.deployments import run_deployment

INGEST_DEPLOYMENT = "ms-ingest-video/ingest"
PAPER_INGEST_DEPLOYMENT = "ms-ingest-paper/ingest"
DECK_INGEST_DEPLOYMENT = "ms-ingest-deck/ingest"


def enqueue_video(video_id: str, user_id: str) -> str:
    """Schedule the ingest flow for one video. Returns the Prefect flow-run id."""
    flow_run = run_deployment(
        name=INGEST_DEPLOYMENT,
        parameters={"video_id": video_id, "user_id": user_id},
        timeout=0,  # fire-and-forget: don't block the API waiting for the run
        flow_run_name=f"ingest-{video_id}",
    )
    return str(flow_run.id)


def enqueue_document(document_id: str, user_id: str, kind: str) -> str:
    """Schedule a paper/deck flow without importing parser dependencies into API."""
    deployments = {"paper": PAPER_INGEST_DEPLOYMENT, "deck": DECK_INGEST_DEPLOYMENT}
    deployment = deployments.get(kind)
    if deployment is None:
        raise ValueError(f"unsupported document kind: {kind}")
    flow_run = run_deployment(
        name=deployment,
        parameters={f"{kind}_id": document_id, "user_id": user_id},
        timeout=0,
        flow_run_name=f"ingest-{document_id}",
    )
    return str(flow_run.id)


def enqueue_source(source_id: str, user_id: str, source: str) -> str:
    """Dispatcher entry point for all manifest source types."""
    if source in {"paper", "deck"}:
        return enqueue_document(source_id, user_id, source)
    return enqueue_video(source_id, user_id)
