"""Ingest worker entrypoint — serves the Prefect flow.

    python -m src.worker

flow.serve() registers the "ms-ingest-video/ingest" deployment in Prefect Cloud
(idempotent) and long-polls for scheduled runs — outbound HTTPS only, no
ports. Scale horizontally by running more replicas of this process; each
executes up to WORKER_CONCURRENCY runs at once.

Sample seeding is NOT done here — it's a one-shot startup gate (seed.py /
src/seeding.py) that the whole stack waits on, so the app never serves a
half-indexed corpus. This worker only handles user uploads + YouTube adds.

Embedding goes to the warm CLIP service when CLIP_SERVICE_URL is set
(docker-compose default); unset, each run loads the model in-process.
"""
import os
import threading
import time

from .db import init_schema
from .ingest.pipeline import ingest_video
from .ingest.paper import ingest_paper
from .ingest.deck import ingest_deck


def _serve_document_flow(flow, name: str, limit: int) -> None:
    """Keep a document deployment registered and polling alongside video."""
    while True:
        try:
            print(f"[worker] serving deployment '{flow.name}/{name}' (concurrency {limit})")
            flow.serve(name=name, limit=limit)
            return
        except KeyboardInterrupt:
            return
        except Exception as exc:
            print(f"[worker] {name} serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


def main():
    init_schema()  # make sure migrations ran before consuming runs
    from .rag import vector_store
    vector_store.ensure_collection()  # up front, not mid-first-ingest
    # Fair scheduler (WFQ): admits pending videos round-robin across users so
    # one bulk uploader can't starve everyone else (src/dispatcher.py).
    from . import dispatcher
    dispatcher.start_in_background()
    limit = int(os.getenv("WORKER_CONCURRENCY", "2"))
    # Each flow has its own named deployment because its parameter differs
    # (video_id, paper_id, deck_id).  They poll in parallel while sharing the
    # manifest-backed fair dispatcher that controls admission.
    for document_flow, name in ((ingest_paper, "ingest"), (ingest_deck, "ingest")):
        threading.Thread(target=_serve_document_flow, args=(document_flow, name, limit),
                         daemon=True, name=f"{document_flow.name}-serve").start()
    # serve() talks to Prefect Cloud on startup; a transient outage (e.g. a 503)
    # used to crash the worker permanently and stop the machine. Self-heal:
    # retry forever so a blip pauses ingest instead of killing the worker.
    while True:
        try:
            print(f"[worker] serving deployment 'ms-ingest-video/ingest' (concurrency {limit})")
            ingest_video.serve(name="ingest", limit=limit)
            break  # clean shutdown
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[worker] serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
