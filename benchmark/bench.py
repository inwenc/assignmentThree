#!/usr/bin/env python3
"""
benchmark/bench.py — the SLA grading gate for Glimpse / Moment-Search-at-Scale.

Proves, against a RUNNING deployment, the five SLAs from the assignment. Exits
NON-ZERO if any threshold is missed (so CI / the grader fails the build).

    Metric                                    Target
    ─────────────────────────────────────────────────────
    POST /admin/documents accept p95          ≤ 300 ms
    Search p95 while a big ingest runs         ≤ 1.3× idle p95
    Cross-source recall@10                     ≥ 0.70
    No-loss under worker crash (--resilience)  100%
    Ingestion throughput (≥2 workers)          ≥ 8 chunks/s

Stdlib only — no pip installs — so it runs anywhere the app is reachable.

USAGE
    export BASE_URL=http://localhost:8100
    export ADMIN_TOKEN=...            # if the server sets ADMIN_TOKEN
    export BENCH_USER=bench           # tenant to isolate benchmark data
    python benchmark/bench.py                     # accept + throughput + backfill + recall
    python benchmark/bench.py --resilience        # kill-a-worker no-loss test
    python benchmark/bench.py --golden benchmark/golden.json

RESILIENCE
    The crash is pluggable — set BENCH_KILL_CMD to a shell command that kills ONE
    worker (e.g. "docker kill ms-worker-1" or "pkill -9 -f worker.py"). Without
    it, --resilience pauses and asks you to kill a worker by hand.

The read/write API shapes match src/api/admin.py + src/api/search.py:
  POST /admin/documents {uri,kind,title} -> 202 {id,status,kind}
  GET  /admin/sources                    -> {sources:[{id,kind,status,pct,chunk_count,...}]}
  POST /api/ask {question,top_k}         -> {citations:[{kind,source_id|video_id,locator,...}]}

⚠️ Endpoint names follow the assignment rubric. If your deploy uses /api/ask vs a
   /ask_stream SSE variant, or /api/documents vs /admin/documents, set the *_PATH
   constants below. The base momentsearch repo uses /api/ask (JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("BASE_URL", "http://localhost:8100").rstrip("/")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
USER = os.getenv("BENCH_USER", "bench")
KILL_CMD = os.getenv("BENCH_KILL_CMD", "")

DOCUMENTS_PATH = "/admin/documents"
SOURCES_PATH = "/admin/sources"
ASK_PATH = "/api/ask"

# ── SLA thresholds (the grading gate) ─────────────────────────────────────────
ACCEPT_P95_MS = 300.0
SEARCH_P95_RATIO = 1.3
RECALL_K = 10
RECALL_MIN = 0.70
THROUGHPUT_MIN = 8.0          # chunks/s
NO_LOSS = 1.0                 # 100%

INDEX_TIMEOUT_S = 900         # how long to wait for a backfill to finish indexing
POLL_S = 2.0


# ── Tiny HTTP client (stdlib) ─────────────────────────────────────────────────
def _req(method: str, path: str, body: dict | None = None,
         auth: bool = False) -> tuple[int, dict]:
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "X-User-Id": USER}
    if auth and ADMIN_TOKEN:
        headers["Authorization"] = f"Bearer {ADMIN_TOKEN}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"_raw": raw.decode(errors="replace")}


def post_document(doc: dict) -> tuple[int, dict, float]:
    """POST one document. Returns (status, body, elapsed_ms)."""
    t0 = time.perf_counter()
    status, body = _req("POST", DOCUMENTS_PATH, doc, auth=True)
    return status, body, (time.perf_counter() - t0) * 1000.0


def get_sources() -> list[dict]:
    _, body = _req("GET", SOURCES_PATH)
    return body.get("sources", [])


def ask(question: str, top_k: int = RECALL_K) -> tuple[dict, float]:
    t0 = time.perf_counter()
    _, body = _req("POST", ASK_PATH, {"question": question, "top_k": top_k})
    return body, (time.perf_counter() - t0) * 1000.0


def retrieve(question: str, top_k: int = RECALL_K) -> tuple[dict, float]:
    """Retrieval only (no LLM). This is what the recall + decoupling SLAs should
    measure — retrieval is milliseconds; the multimodal LLM (seconds) is a
    separate concern and would swamp the signal."""
    t0 = time.perf_counter()
    _, body = _req("POST", "/api/retrieve", {"question": question, "top_k": top_k})
    return body, (time.perf_counter() - t0) * 1000.0


def p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    # nearest-rank p95
    k = max(0, min(len(xs) - 1, int(round(0.95 * (len(xs) - 1)))))
    return xs[k]


# ── Golden set ────────────────────────────────────────────────────────────────
def load_golden(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"FATAL: golden set not found at {path}. See benchmark/golden.example.json — "
                 "never fabricate recall; use real labeled queries.")
    g = json.loads(p.read_text(encoding="utf-8"))
    if not g.get("documents") or not g.get("queries"):
        sys.exit("FATAL: golden set needs non-empty 'documents' and 'queries'.")
    return g


# ── Ingest + wait helpers ─────────────────────────────────────────────────────
def ingest_docs(docs: list[dict]) -> tuple[list[str], list[float]]:
    """POST each document. Returns (ids, accept_latencies_ms)."""
    ids, lat = [], []
    for d in docs:
        status, body, ms = post_document(d)
        if status != 202:
            sys.exit(f"FATAL: /admin/documents returned {status}: {body}")
        ids.append(body["id"])
        lat.append(ms)
    return ids, lat


def wait_indexed(ids: set[str], timeout_s: int = INDEX_TIMEOUT_S) -> dict[str, dict]:
    """Poll /admin/sources until every id is 'indexed' or 'failed' (or timeout).
    Returns {id: source_row}."""
    deadline = time.time() + timeout_s
    seen: dict[str, dict] = {}
    while time.time() < deadline:
        rows = {s["id"]: s for s in get_sources() if s["id"] in ids}
        seen.update(rows)
        pending = [i for i in ids if seen.get(i, {}).get("status") not in ("indexed", "failed")]
        if not pending:
            return seen
        time.sleep(POLL_S)
    return seen


# ── The five checks ───────────────────────────────────────────────────────────
class Gate:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str):
        self.results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")

    def exit(self):
        print("\n" + "=" * 64)
        failed = [n for n, ok, _ in self.results if not ok]
        if failed:
            print(f"SLA GATE FAILED — {len(failed)}/{len(self.results)} missed: {', '.join(failed)}")
            sys.exit(1)
        print(f"SLA GATE PASSED — {len(self.results)}/{len(self.results)} met.")
        sys.exit(0)


def run_standard(g: dict, gate: Gate) -> None:
    docs = g["documents"]
    queries = g["queries"]

    # 1) ACCEPT LATENCY + ingest the golden corpus ----------------------------
    print("\n▶ Accept latency + corpus ingest")
    ids, accept_lat = ingest_docs(docs)
    title_by_id = {i: d.get("title") for i, d in zip(ids, docs)}
    ap95 = p95(accept_lat)
    gate.check("accept_p95", ap95 <= ACCEPT_P95_MS,
               f"p95={ap95:.1f}ms (target ≤ {ACCEPT_P95_MS:.0f}ms, n={len(accept_lat)})")

    # 2) THROUGHPUT — wait for indexing, measure chunks/s ---------------------
    print("\n▶ Throughput (waiting for indexing)")
    t0 = time.time()
    rows = wait_indexed(set(ids))
    elapsed = max(1e-6, time.time() - t0)
    indexed = [r for r in rows.values() if r.get("status") == "indexed"]
    total_chunks = sum(int(r.get("chunk_count") or 0) for r in indexed)
    tput = total_chunks / elapsed
    gate.check("throughput", tput >= THROUGHPUT_MIN,
               f"{tput:.1f} chunks/s over {elapsed:.0f}s ({total_chunks} chunks, "
               f"{len(indexed)}/{len(ids)} indexed; ensure ≥2 workers)")

    # 3) RECALL@10 (cross-source) ---------------------------------------------
    print("\n▶ Cross-source recall@10")
    hits = 0
    for q in queries:
        body, _ = retrieve(q["question"], top_k=RECALL_K)   # LLM-free: citations only
        cites = body.get("citations", [])[:RECALL_K]
        want_title = q.get("expect_title")
        got = any(title_by_id.get(c.get("source_id") or c.get("video_id")) == want_title
                  or (c.get("title") == want_title) for c in cites)
        hits += 1 if got else 0
        if not got:
            print(f"     miss: {q['question']!r} — expected {want_title!r} not in top-{RECALL_K}")
    recall = hits / len(queries)
    gate.check("recall@10", recall >= RECALL_MIN,
               f"{recall:.2f} ({hits}/{len(queries)}) (target ≥ {RECALL_MIN})")

    # 4) DECOUPLING — RETRIEVAL p95 stays flat during a big backfill -----------
    #    (retrieval only — the LLM synthesis is a separate, seconds-long concern)
    print("\n▶ Decoupling: retrieval p95 idle vs. during backfill")
    idle = [retrieve(q["question"])[1] for q in queries for _ in range(3)]
    idle_p95 = p95(idle)

    # kick off a big backfill in the background (duplicate the golden docs)
    backfill = docs * max(1, g.get("backfill_multiplier", 5))
    stop = threading.Event()

    def _backfill():
        for d in backfill:
            if stop.is_set():
                return
            post_document(d)

    bt = threading.Thread(target=_backfill, daemon=True)
    bt.start()
    time.sleep(1.0)  # let the ingest spike ramp
    during = [retrieve(q["question"])[1] for _ in range(4) for q in queries]
    stop.set()
    during_p95 = p95(during)
    ratio = during_p95 / idle_p95 if idle_p95 else float("inf")
    gate.check("search_p95_under_load", ratio <= SEARCH_P95_RATIO,
               f"idle={idle_p95:.0f}ms during={during_p95:.0f}ms ratio={ratio:.2f} "
               f"(target ≤ {SEARCH_P95_RATIO})")


def run_resilience(g: dict, gate: Gate) -> None:
    """Kill a worker mid-ingest → 0 dropped, all resume to indexed, finished
    stages not re-run (idempotent point ids make re-runs no-ops)."""
    print("\n▶ Resilience: no-loss under worker crash")
    docs = g["documents"] * max(2, g.get("resilience_multiplier", 3))
    ids, _ = ingest_docs(docs)
    idset = set(ids)

    # wait until at least one is mid-flight (not pending, not indexed)
    print("  waiting for jobs to go in-flight…")
    inflight_seen = False
    deadline = time.time() + 120
    while time.time() < deadline:
        rows = {s["id"]: s for s in get_sources() if s["id"] in idset}
        mid = [i for i, r in rows.items()
               if r.get("status") in ("parsing", "chunking", "embedding")]
        if mid:
            inflight_seen = True
            break
        if all(rows.get(i, {}).get("status") == "indexed" for i in idset):
            break
        time.sleep(0.5)

    # kill a worker
    if KILL_CMD:
        print(f"  killing a worker: {KILL_CMD}")
        subprocess.run(KILL_CMD, shell=True, check=False)
    else:
        input(f"  >>> Kill ONE worker now (mid-ingest), then press Enter "
              f"(in-flight seen: {inflight_seen}) …")

    # everything must still reach indexed
    print("  asserting every source resumes to 'indexed'…")
    rows = wait_indexed(idset, timeout_s=int(os.getenv("RESILIENCE_WAIT_S", "240")))
    indexed = [i for i in idset if rows.get(i, {}).get("status") == "indexed"]
    lost = [i for i in idset if rows.get(i, {}).get("status") != "indexed"]
    frac = len(indexed) / len(idset)
    gate.check("no_loss", frac >= NO_LOSS,
               f"{len(indexed)}/{len(idset)} resumed to indexed"
               + (f"; LOST: {lost}" if lost else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Moment-Search-at-Scale SLA gate")
    ap.add_argument("--resilience", action="store_true",
                    help="run the kill-a-worker no-loss test instead of the SLA suite")
    ap.add_argument("--golden", default=os.getenv("BENCH_GOLDEN", "benchmark/golden.json"))
    args = ap.parse_args()

    # health check first — a clear message beats a wall of connection errors
    try:
        status, _ = _req("GET", "/api/health")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"FATAL: cannot reach {BASE_URL} ({e}). Is the app up?")
    print(f"Target {BASE_URL} (user={USER!r})  health={status}")

    g = load_golden(args.golden)
    gate = Gate()
    if args.resilience:
        run_resilience(g, gate)
    else:
        run_standard(g, gate)
    gate.exit()


if __name__ == "__main__":
    main()