"""Prefect Cloud trigger layer — the API schedules runs, workers execute them.

Video, paper, and deck flows each have a named ingest deployment. The API
never imports parser/model dependencies; it asks Prefect to schedule the
appropriate source flow and returns immediately. Retries/backoff live on each
flow's tasks and failures remain visible/retryable in the Prefect UI.
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
