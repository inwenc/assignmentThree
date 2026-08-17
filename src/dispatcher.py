"""Fair dispatcher — the WFQ scheduler that sits in front of Prefect.

Why this exists: if the API enqueued every source to Prefect at register-time,
Prefect would run them in submitted order (FIFO) — one user who uploads 50
videos blocks everyone behind them. Instead, videos wait `pending` in Postgres
and THIS loop admits them:

  every DISPATCH_INTERVAL_S:
    slots = DISPATCH_MAX_INFLIGHT - (videos currently queued/running)
    claim up to `slots` pending videos in FAIR order (round-robin across users)
    schedule the matching video, paper, or deck Prefect run

Because only ~capacity videos are ever handed to Prefect at once, the *waiting
line lives in our DB, fairly ordered* (db.wfq_claim) rather than FIFO inside
Prefect. No user can starve the others. Set ENABLE_FAIR_DISPATCH=false to fall
back to immediate FIFO enqueue (useful for A/B teaching the difference).

Runs as a background thread in worker.py. With one worker that's exact; with
several, each runs a dispatcher — the atomic claim keeps videos handed out once,
at worst mildly over-admitting (harmless; Prefect still caps execution).
"""
from __future__ import annotations

import threading
import time

from . import config, db, jobs


def dispatch_once() -> int:
    """Admit as many fairly-chosen pending sources as free capacity allows.
    Returns how many were dispatched this tick."""
    slots = config.DISPATCH_MAX_INFLIGHT - db.count_inflight()
    if slots <= 0:
        return 0
    claimed = db.wfq_claim(slots)
    for row in claimed:
        try:
            # Paper and deck rows use the same fair queue but their own Prefect
            # deployments/parameters; legacy sources retain the video flow.
            jobs.enqueue_source(row["id"], row["user_id"], row.get("source", "upload"))
        except Exception as exc:
            # Couldn't reach Prefect — put it back so it's retried next tick.
            db.set_status(row["id"], "pending", error=f"dispatch: {exc}")
    if claimed:
        print(f"[dispatch] admitted {len(claimed)} source(s) "
              f"({db.count_inflight()}/{config.DISPATCH_MAX_INFLIGHT} in flight)")
    return len(claimed)


def run_forever() -> None:
    print(f"[dispatch] fair scheduler on — max in-flight "
          f"{config.DISPATCH_MAX_INFLIGHT}, tick {config.DISPATCH_INTERVAL_S}s")
    while True:
        try:
            dispatch_once()
        except Exception as exc:  # never let the scheduler thread die
            print(f"[dispatch] error: {type(exc).__name__}: {exc}")
        time.sleep(config.DISPATCH_INTERVAL_S)


def start_in_background() -> None:
    """Start the dispatcher as a daemon thread (no-op if fair dispatch is off)."""
    if not config.ENABLE_FAIR_DISPATCH:
        print("[dispatch] fair dispatch disabled — FIFO (immediate enqueue)")
        return
    threading.Thread(target=run_forever, daemon=True, name="dispatcher").start()
