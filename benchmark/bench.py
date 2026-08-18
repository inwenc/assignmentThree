#!/usr/bin/env python3
"""Benchmark + SLA gate for Assignment 3 — Moment Search at Scale.

    python benchmark/bench.py                 # accept-latency, ingest-vs-search, recall
    python benchmark/bench.py --resilience    # kill a worker mid-ingest, assert no loss
    python benchmark/bench.py --json out.json # also write machine-readable results

Exits non-zero if ANY target in sla.json is missed, so it doubles as your grading
gate and a CI check.

This is a SCAFFOLD. The measurement skeleton, the SLA comparison, and the exit
code are done. You fill the four TODOs so it measures YOUR running app:
labeled queries for recall, the concurrent-ingest load, the throughput probe,
and the worker-crash step. Keep the gates in sla.json as-is.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SLA = json.loads((ROOT / "benchmark" / "sla.json").read_text())
BASE = os.getenv("BASE_URL", "http://localhost:8100").rstrip("/")
ADMIN = os.getenv("ADMIN_TOKEN", "")

TERMINAL = ("indexed", "skipped", "failed")
QUERIES = ROOT / "benchmark" / "queries.jsonl"

# The backfill the load-bearing gates run against: real documents, fetched over
# the network, parsed and embedded like any other. Override with
# BENCH_CORPUS=<comma-separated urls> to point the benchmark at your own set.
#
# SIZED, not picked. The first eight below are the retrieval papers this product
# is built on and they used to be the whole list. Measured against them, both
# load-bearing gates were unmeasurable rather than failed:
#
#   * search_p95_during_ingest_ratio compares search latency on a quiet system
#     against search latency during the backfill. Eight short papers drained in
#     59s while the "during" measurement — real /ask_stream calls at ~5s each —
#     needed about three minutes, so the second half ran on a system that was
#     already idle again. bench.py catches this itself (`load_held`) and fails
#     the gate rather than reporting the ~1.0 ratio two idle systems produce.
#   * ingest_throughput_chunks_per_s then measured its own wall clock from
#     registration to the moment it FIRST looked, which was after that idle
#     measurement had run its course: 394 chunks over a reported 181s, where the
#     worker logs put the real window at 59.4s. 2.17 chunks/s reported, 6.6 real.
#
# So the corpus has to outlast the measurement standing next to it. Second lever,
# from the same logs: per-document cost dominates. 20 chunks took 20.1s and 78
# chunks took 23.2s, so chunks/s here mostly reports how many chunks are in an
# average document, not how fast the system is. The eight short papers averaged
# 49 chunks. The twelve added below are the long-form model reports and surveys
# in the same lineage — same subject, an order of magnitude more pages.
#
# Say the quiet part out loud, because the two are easy to confuse: enlarging the
# corpus makes both gates MEASURE something, and it also raises chunks/s without
# the system getting faster. PRODUCT_EVAL.md has to state that chunks/s is a
# function of document size, or the number reads as a speed it is not.
#
# Deliberately NOT here: arxiv.org/pdf/2108.07258 (On the Opportunities and Risks
# of Foundation Models, ~200pp). It is the obvious stress test and that is the
# problem — the `parsing` budget is a flat 300s with no per-page term
# (src/config.py), so it is the one document that might be reaped mid-run, and a
# reaped-then-retried row would inflate exactly the window being measured. Add it
# as its own observation, not inside the run you need to be readable.
#
# Also NOT here, and for a sharper reason: BENCH_VICTIM (2303.18223, see
# resilience()). A document's id is derived from its normalized uri, so a victim
# that is already in the backfill is the SAME row — resilience() would then kill
# a worker over one of the eight it just registered instead of over the large
# source it registers deliberately, and its "wait until the victim is mid-
# embedding" step would be watching the wrong thing. Keep the two lists disjoint.
CORPUS = [u for u in os.getenv("BENCH_CORPUS", "").split(",") if u.strip()] or [
    # The method papers: short, dense, 6-21 pages each.
    "https://arxiv.org/pdf/1706.03762",   # Attention Is All You Need
    "https://arxiv.org/pdf/1810.04805",   # BERT
    "https://arxiv.org/pdf/1908.10084",   # Sentence-BERT
    "https://arxiv.org/pdf/2005.11401",   # RAG
    "https://arxiv.org/pdf/2004.04906",   # DPR
    "https://arxiv.org/pdf/2007.01282",   # FiD
    "https://arxiv.org/pdf/2112.09118",   # Contriever
    "https://arxiv.org/pdf/2212.10496",   # HyDE
    # Long-form: model reports, benchmarks and surveys from the same literature.
    "https://arxiv.org/pdf/1910.10683",   # T5
    "https://arxiv.org/pdf/2005.14165",   # GPT-3
    "https://arxiv.org/pdf/2101.00027",   # The Pile
    "https://arxiv.org/pdf/2104.08663",   # BEIR
    "https://arxiv.org/pdf/2201.11903",   # Chain-of-Thought prompting
    "https://arxiv.org/pdf/2203.02155",   # InstructGPT
    "https://arxiv.org/pdf/2211.05100",   # BLOOM
    "https://arxiv.org/pdf/2210.11416",   # Scaling Instruction-Finetuned LMs
    "https://arxiv.org/pdf/2302.13971",   # LLaMA
    "https://arxiv.org/pdf/2307.09288",   # Llama 2
    "https://arxiv.org/pdf/2312.10997",   # Retrieval-Augmented Generation: a survey
    "https://arxiv.org/pdf/2402.19473",   # RAG for AI-Generated Content: a survey
]

# How the worker is taken down and brought back for --resilience. `kill` is a
# SIGKILL, which is the point: the process gets no chance to tidy up, exactly
# like a machine going away. `start` revives the same container rather than
# recreating it, so it keeps the environment it was created with — `up` would
# re-run the seed gate and, with secrets injected at compose time rather than
# read from a file, that gate fails and the worker never returns.
WORKER_KILL = os.getenv("WORKER_KILL_CMD", "docker compose kill worker")
WORKER_START = os.getenv("WORKER_START_CMD", "docker compose start worker")


def _req(method, path, body=None, token=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        return 0, str(e), (time.perf_counter() - t0) * 1000


def p95(xs):
    return statistics.quantiles(xs, n=100)[94] if len(xs) >= 20 else (max(xs) if xs else 0.0)


def measure_accept_latency(n=30):
    """POST /admin/documents should enqueue-and-return fast (no parsing in-request).

    The probe URLs do not resolve, on purpose: the point is that registration
    never touches the source. But that leaves 30 rows the queue will now chew on
    with retry backoff, and with DISPATCH_MAX_INFLIGHT=2 they occupy the workers
    for the better part of an hour — right through the throughput and
    decoupling measurements below. So they are removed again once measured.
    Without that, the two gates after this one are measuring a poisoned system.
    """
    lat, ids = [], []
    for i in range(n):
        st, body, ms = _req("POST", "/admin/documents", token=ADMIN,
                            body={"uri": f"https://example.com/probe_{i}.pdf",
                                  "kind": "paper", "title": f"probe {i}"})
        if st == 202:
            lat.append(ms)
            try:
                ids.append(json.loads(body)["id"])
            except Exception:  # noqa: BLE001 — a 202 without an id is the gate's problem, not ours
                pass
    for pid in ids:
        _req("DELETE", f"/api/videos/{pid}", token=ADMIN)
    return p95(lat) if lat else float("inf")


def measure_search_p95(n=40):
    q = "what does the survey say about hybrid retrieval"
    lat = []
    for _ in range(n):
        st, _, ms = _req("GET", "/ask_stream?q=" + urllib.parse.quote(q))
        if st == 200:
            lat.append(ms)
    return p95(lat) if lat else float("inf")


# ── Talking to the manifest ──────────────────────────────────────────────────

def _sources():
    st, body, _ = _req("GET", "/admin/sources", token=ADMIN)
    if st != 200:
        return []
    try:
        return json.loads(body).get("sources", [])
    except Exception:  # noqa: BLE001
        return []


def _register(uri, kind="paper", title=None):
    """Register one document; returns its id, or None if it was refused."""
    st, body, _ = _req("POST", "/admin/documents", token=ADMIN,
                       body={"uri": uri, "kind": kind, "title": title or uri.rsplit("/", 1)[-1]})
    if st != 202:
        print(f"  ! register {uri} -> {st} {body[:120]}")
        return None
    try:
        return json.loads(body)["id"]
    except Exception:  # noqa: BLE001
        return None


def _states(ids):
    by_id = {s["id"]: s for s in _sources()}
    return {i: by_id.get(i, {}).get("status") for i in ids}


def _wait_terminal(ids, timeout_s, label="waiting"):
    """Block until every id is finished (or the clock runs out). Returns the
    final states — the caller decides whether that is a pass.

    Prints a line every 15s. A silent wait of up to half an hour is
    indistinguishable from a hang, and whoever runs this benchmark has no way
    to tell the difference from the outside.
    """
    end = time.time() + timeout_s
    t0, last = time.time(), 0.0
    while time.time() < end:
        states = _states(ids)
        if all(s in TERMINAL for s in states.values()):
            return states
        if time.time() - last >= 15:
            last = time.time()
            tally = {}
            for s in states.values():
                tally[s] = tally.get(s, 0) + 1
            print(f"  [{label} {time.time()-t0:.0f}s] "
                  + " ".join(f"{k}={v}" for k, v in sorted(tally.items(), key=lambda x: str(x[0]))))
        time.sleep(3)
    print(f"  ! gave up after {timeout_s}s")
    return _states(ids)


def measure_recall_at_10():
    """Recall@10 over the labeled set: is the right source in the top ten?

    Returns (recall, checked). A missing or empty query file returns (0.0, 0) —
    the gate then fails, which is the honest outcome: an unmeasured metric is
    not a passing one, and the alternative (skip the gate) is how a benchmark
    quietly starts reporting on nothing.

    One line per query in benchmark/queries.jsonl:
        {"q": "...", "kind": "paper", "source_id": "doc_ab12", "page": 4}
    `kind` alone is a weak label and is accepted, but a source_id is what makes
    this measure retrieval rather than luck. A locator (page/slide/ms) is
    checked when given, with a one-page tolerance because a claim that straddles
    a page break is a real answer, not a miss.
    """
    if not QUERIES.exists():
        print(f"  ! {QUERIES} is missing — recall cannot be measured")
        return 0.0, 0
    labeled = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    if not labeled:
        return 0.0, 0
    hits = 0
    for item in labeled:
        cites = (_sse_citations(item["q"]) or [])[:10]
        if any(_matches(c, item) for c in cites):
            hits += 1
        else:
            print(f"  miss: {item['q'][:60]!r}")
    return hits / len(labeled), len(labeled)


def _matches(citation, want):
    if want.get("source_id") and citation.get("sourceId") != want["source_id"]:
        return False
    if want.get("kind") and citation.get("kind") != want["kind"]:
        return False
    loc = citation.get("locator") or {}
    if want.get("page") is not None and abs((loc.get("page") or -99) - want["page"]) > 1:
        return False
    if want.get("slide") is not None and loc.get("slide") != want["slide"]:
        return False
    return True


def _sse_citations(q, timeout=90):
    """Read /ask_stream until the citations frame. Same contract eval.py uses:
    one `data:` line carrying a JSON object with `citations` at the top level."""
    url = f"{BASE}/ask_stream?q=" + urllib.parse.quote(q)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                d = json.loads(line[5:].strip())
                if "citations" in (d.get("detail") or d):
                    return (d.get("detail") or d)["citations"]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! /ask_stream failed: {type(exc).__name__}: {exc}")
    return None


class _QueueMonitor(threading.Thread):
    """Watches a backfill from the moment it is registered.

    measure_throughput used to sample the queue itself, and it is only reached
    AFTER the search-under-load gate has finished — which on any corpus that
    drains quickly is after the queue is already empty. Two numbers came out
    wrong for the one reason: `peak` reported 0 because nothing was ever in
    flight when it first looked, and `elapsed` ran to that first look rather
    than to the last source finishing, charging the previous gate's runtime to
    ingest. Measured on the eight-paper corpus: a 59.4s ingest window reported
    as 181s, 6.6 chunks/s reported as 2.17.

    So the sampling now starts where the clock starts. Daemon, because a
    benchmark that hangs on its own instrumentation is worse than one that
    loses a sample.
    """

    def __init__(self, ids, t0, interval_s=2.0, timeout_s=1800):
        super().__init__(daemon=True, name="queue-monitor")
        self.ids, self.t0 = ids, t0
        self.interval_s, self.deadline = interval_s, t0 + timeout_s
        self.peak = 0
        self.finished_at = None      # None: never observed all-terminal
        self.states = {i: None for i in ids}

    def run(self):
        while time.time() < self.deadline:
            try:
                self.states = _states(self.ids)
            except Exception as exc:  # noqa: BLE001
                # The thread must outlive a blip. If it died here instead,
                # finished_at would stay None, and measure_throughput would sit
                # out its whole timeout waiting for a sample that never comes —
                # a transient 502 turning into a 15-minute hang.
                print(f"  ! queue monitor: {type(exc).__name__}: {exc}")
                time.sleep(self.interval_s)
                continue
            self.peak = max(self.peak, sum(
                1 for s in self.states.values()
                if s not in TERMINAL and s != "pending"))
            if self.states and all(s in TERMINAL for s in self.states.values()):
                self.finished_at = time.time()
                return
            time.sleep(self.interval_s)


def run_backfill(uris, label):
    """Register a batch and return (ids, started_at, monitor). Registration is
    the cheap half — these return in milliseconds and the work happens on the
    queue, which is what the monitor is there to watch."""
    print(f"[{label}] registering {len(uris)} documents…")
    t0 = time.time()
    ids = [i for i in (_register(u) for u in uris) if i]
    print(f"[{label}] {len(ids)} accepted")
    monitor = _QueueMonitor(ids, t0)
    monitor.start()
    return ids, t0, monitor


def measure_throughput(monitor, timeout_s=900):
    """Chunks per second across a backfill, wall clock from first register to
    the last source reaching a terminal state.

    Both ends of that clock come from the monitor run_backfill started, not from
    this function: it is called after the gate before it and would otherwise be
    timing that gate as well as the ingest (see _QueueMonitor). This blocks only
    for whatever is STILL running by the time it is reached, which on a corpus
    large enough for the ratio gate is the normal case.

    Counts only what actually landed: `frame_count` doubles as the chunk count
    for documents, and a source that failed contributes neither its chunks nor
    an excuse.
    """
    ids = monitor.ids
    end, last = time.time() + timeout_s, 0.0
    while monitor.finished_at is None and time.time() < end:
        if time.time() - last >= 15:
            last = time.time()
            done = sum(1 for s in monitor.states.values() if s in TERMINAL)
            print(f"  [backfill {time.time()-monitor.t0:.0f}s] {done}/{len(ids)} "
                  f"finished, {monitor.peak} concurrent at peak")
        time.sleep(1)
    # Whether the ingest outlasted the gate before this one is not a detail: it
    # is exactly what decides whether search_p95_during_ingest_ratio measured a
    # loaded system or two idle ones. Printed either way, so a reader of the log
    # can tell which of the two happened without reconstructing it from
    # timestamps.
    if monitor.finished_at:
        drained = monitor.finished_at - monitor.t0
        slack = time.time() - monitor.finished_at
        print(f"  [backfill] drained {drained:.0f}s after registration, "
              f"{slack:.0f}s before this gate was reached"
              + (" — the ratio gate ran partly on an idle system"
                 if slack > 5 else ""))
    states, elapsed = monitor.states, (
        (monitor.finished_at or time.time()) - monitor.t0)
    peak = monitor.peak
    by_id = {s["id"]: s for s in _sources()}
    chunks = sum((by_id.get(i, {}).get("frame_count") or 0)
                 for i, st in states.items() if st == "indexed")
    stuck = {i: st for i, st in states.items() if st not in TERMINAL}
    if stuck:
        print(f"  ! {len(stuck)} source(s) never finished: {stuck}")
    # The SLA reads "≥2 workers, warm", so the number is only interpretable
    # next to the capacity it was measured under. `peak` is what actually ran at
    # once, which is min(DISPATCH_MAX_INFLIGHT, worker replicas x concurrency) —
    # measured rather than read off the config, because those two can disagree.
    print(f"  {chunks} chunks over {elapsed:.0f}s "
          f"({sum(1 for s in states.values() if s == 'indexed')}/{len(ids)} indexed, "
          f"at most {peak} running at once)")
    return (chunks / elapsed) if elapsed > 0 else 0.0


def resilience():
    """Kill a worker mid-ingest and assert nothing is lost.

    The kill has to LAND, and that is harder than it looks: a small document is
    warm-ingested in about ten seconds, so "wait until something is not
    terminal, then kill" reliably kills an idle worker and proves nothing. So
    one deliberately large source is registered alongside the backfill and the
    kill waits for THAT one to be visibly mid-embedding.

    Three claims, each with its own evidence and its own failure message,
    because they fail for different reasons:
      1. the crash landed at all — the reaper's "[reap] <id> … requeued" line,
         which only exists if a live run was lost and re-offered;
      2. every source reaches a terminal state, and none of them `failed`;
      3. a stage that had already finished was not re-run — the worker's
         "[fetch] <id>: reusing <file>" line, which only appears when the
         resumed run finds the completed download already on disk.
    Without (1), (2) and (3) are answers to a question nobody asked.
    """
    victim_uri = os.getenv("BENCH_VICTIM", "https://arxiv.org/pdf/2303.18223")
    ids, _, _ = run_backfill(CORPUS, "resilience")
    victim = _register(victim_uri, title="resilience victim (large)")
    if not ids or not victim:
        print("  ! nothing registered — cannot test")
        return False
    ids = ids + [victim]

    print(f"[resilience] waiting for {victim} to be mid-embedding…")
    caught = None
    for _ in range(300):
        s = {x["id"]: x for x in _sources()}.get(victim, {})
        pct = s.get("pct") or 0
        if s.get("status") == "embedding" and 15 <= pct <= 85:
            caught = f"{s['status']} {pct}%"
            break
        if s.get("status") in TERMINAL:
            print(f"  ! {victim} finished before the kill could land ({s.get('status')}); "
                  "use a larger BENCH_VICTIM")
            return False
        time.sleep(2)
    if not caught:
        print("  ! never observed the victim mid-embedding — cannot test a crash")
        return False

    print(f"[resilience] killing the worker with {victim} at {caught}")
    subprocess.run(WORKER_KILL.split(), cwd=ROOT, capture_output=True)
    time.sleep(5)
    subprocess.run(WORKER_START.split(), cwd=ROOT, capture_output=True)
    print(f"[resilience] worker restarted; nobody touches the queue from here on")

    states = _wait_terminal(ids, 1800, "recovering")
    lost = {i: s for i, s in states.items() if s not in TERMINAL}
    failed = {i: s for i, s in states.items() if s == "failed"}
    log = _worker_log()
    requeued = f"[reap] {victim}" in log
    reused = f"[fetch] {victim}: reusing" in log

    print(f"[resilience] terminal={len(states)-len(lost)}/{len(ids)} "
          f"lost={len(lost)} failed={len(failed)} "
          f"requeued-by-reaper={requeued} finished-stage-reused={reused}")
    if not requeued:
        print("  ! no '[reap] …' line for the victim: the crash did not interrupt it, so "
              "this run tested nothing. (Also check PYTHONUNBUFFERED — a buffered "
              "worker logs nothing at all.)")
    if lost:
        print(f"  ! stuck, never finished: {lost}")
    if failed:
        print(f"  ! ended failed: {failed}")
    if requeued and not reused:
        print("  ! requeued but no '[fetch] … reusing …': the resumed run re-downloaded "
              "a stage it had already completed")
    return requeued and reused and not lost and not failed


def _worker_log(lines=600):
    out = subprocess.run(f"docker compose logs worker --tail {lines}".split(),
                         cwd=ROOT, capture_output=True, text=True)
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resilience", action="store_true")
    ap.add_argument("--json", dest="json_out", default="")
    args = ap.parse_args()

    results, failures = {}, []

    def gate(name, value, ok, target):
        results[name] = {"value": value, "target": target, "pass": bool(ok)}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value} (target {target})")
        if not ok:
            failures.append(name)

    if args.resilience:
        no_loss = resilience()
        gate("no_loss_under_crash", no_loss, no_loss and SLA["no_loss_required"], "0 dropped, all indexed")
        return sys.exit(1 if failures else 0)

    # 1. accept latency
    a = measure_accept_latency()
    gate("accept_latency_p95_ms", round(a, 1), a <= SLA["accept_latency_p95_ms"], SLA["accept_latency_p95_ms"])

    # 2. search stays fast during a big ingest.
    #
    # The idle number is measured first, on a quiet system. Then the backfill is
    # registered and the SAME measurement runs again while it drains — the load
    # has to be real, so this is checked rather than assumed: if the queue
    # emptied before the second measurement finished, the ratio would be two
    # idle numbers divided by each other, which is ~1.0 and means nothing. That
    # is the way this gate goes green while proving no decoupling at all.
    print("[idle] measuring search p95 on a quiet system…")
    idle = measure_search_p95()
    ids, _, monitor = run_backfill(CORPUS, "backfill")
    busy_at_start = sum(1 for s in _states(ids).values() if s not in TERMINAL)
    print(f"[during] measuring search p95 with {busy_at_start} source(s) in the queue…")
    during = measure_search_p95()
    busy_at_end = sum(1 for s in _states(ids).values() if s not in TERMINAL)
    ratio = (during / idle) if idle else float("inf")
    print(f"  idle p95 {idle:.0f}ms · during {during:.0f}ms · "
          f"queue {busy_at_start} -> {busy_at_end} unfinished")
    load_held = busy_at_end > 0
    if not load_held:
        print("  ! the backfill finished before the measurement did — this ratio "
              "compares two idle systems and proves nothing")
    gate("search_p95_during_ingest_ratio", round(ratio, 2),
         load_held and ratio <= SLA["search_p95_during_ingest_ratio_max"],
         SLA["search_p95_during_ingest_ratio_max"])

    # 3. ingestion throughput — the same backfill, timed to completion.
    throughput = measure_throughput(monitor)
    gate("ingest_throughput_chunks_per_s", round(throughput, 2),
         throughput >= SLA["ingest_throughput_min_chunks_per_s"], SLA["ingest_throughput_min_chunks_per_s"])

    # 4. recall@10 on labeled queries, now that the corpus is indexed
    recall, checked = measure_recall_at_10()
    gate("recall_at_10", round(recall, 3), recall >= SLA["recall_at_10_min"],
         f"{SLA['recall_at_10_min']} over {checked} labeled queries")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")

    print(f"\n{'ALL SLAs PASS' if not failures else 'SLA FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()