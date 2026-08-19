#!/usr/bin/env python3
"""Benchmark + SLA gate for Assignment 3 — Moment Search at Scale.

    python benchmark/bench.py                 # accept-latency, ingest-vs-search, recall
    python benchmark/bench.py --resilience    # kill a worker mid-ingest, assert no loss
    python benchmark/bench.py --json out.json # also write machine-readable results

Exits non-zero if ANY target in sla.json is missed, so it doubles as your grading
gate and a CI check.

Fixtures are JSONL so the benchmark measures a real, reproducible corpus:

  documents.jsonl: {"uri": "...pdf", "kind": "paper", "title": "..."}
  queries.jsonl:   {"query": "...", "expected": [{"title": "...",
                    "kind": "paper", "locator": {"page": 4}}]}

The document fixture path may also be supplied with --documents. Worker crash
and restart commands are intentionally explicit environment variables; the
benchmark never guesses how your deployment is managed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
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
BASE = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
ADMIN = os.getenv("ADMIN_TOKEN", "")
USER = os.getenv("BENCH_USER", "benchmark")


def _req(method, path, body=None, token=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    req.add_header("x-user-id", USER)
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


def _json(text, context):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} did not return JSON: {text[:300]}") from exc


def load_jsonl(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing benchmark fixture: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_no}: each line must be a JSON object")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"Benchmark fixture is empty: {path}")
    return rows


def request_document(doc):
    required = {"uri", "kind"}
    missing = required - doc.keys()
    if missing:
        raise RuntimeError(f"Document fixture missing {', '.join(sorted(missing))}: {doc}")
    payload = {key: doc[key] for key in ("uri", "kind", "title") if doc.get(key)}
    st, text, ms = _req("POST", "/admin/documents", body=payload, token=ADMIN)
    if st != 202:
        raise RuntimeError(f"Document registration failed ({st}): {text[:300]}")
    response = _json(text, "POST /admin/documents")
    if not response.get("id"):
        raise RuntimeError("POST /admin/documents returned no source id")
    return response["id"], ms


def delete_sources(source_ids):
    """Remove accept-latency probes without delaying the benchmark.

    Deletion may wait on a remote Qdrant delete. It is housekeeping only, so
    each request runs in a bounded daemon thread rather than turning the
    accept-latency check into thirty sequential delete waits.
    """
    def delete_one(source_id):
        _req("DELETE", f"/api/videos/{urllib.parse.quote(source_id)}",
             token=ADMIN, timeout=3)

    for source_id in source_ids:
        threading.Thread(target=delete_one, args=(source_id,), daemon=True).start()


def measure_accept_latency(n=30):
    """POST /admin/documents should enqueue-and-return fast (no parsing in-request)."""
    lat = []
    probe_ids = []
    for i in range(n):
        st, text, ms = _req("POST", "/admin/documents", token=ADMIN,
                         body={"uri": f"https://example.com/probe_{i}.pdf",
                               "kind": "paper", "title": f"probe {i}"})
        if st == 202:
            lat.append(ms)
            try:
                probe_ids.append(_json(text, "accept-latency probe")["id"])
            except (KeyError, RuntimeError):
                pass
    delete_sources(probe_ids)
    return p95(lat) if lat else float("inf")


def measure_search_p95(n=40):
    q = "what does the survey say about hybrid retrieval"
    lat = []
    for _ in range(n):
        st, _, ms = _req("POST", "/api/ask", body={"question": q, "top_k": 10})
        if st == 200:
            lat.append(ms)
    return p95(lat) if lat else float("inf")


def sources_by_id():
    st, text, _ = _req("GET", "/admin/sources", token=ADMIN)
    if st != 200:
        raise RuntimeError(f"GET /admin/sources failed ({st}): {text[:300]}")
    return {row["id"]: row for row in _json(text, "GET /admin/sources").get("sources", [])}


def wait_for_terminal(source_ids, timeout_s, poll_s):
    """Wait for all sources to finish and return their final manifest rows."""
    deadline = time.monotonic() + timeout_s
    wanted = set(source_ids)
    while time.monotonic() < deadline:
        rows = sources_by_id()
        observed = {source_id: rows.get(source_id) for source_id in wanted}
        if all(row and row.get("status") in {"indexed", "failed", "skipped"}
               for row in observed.values()):
            return observed
        time.sleep(poll_s)
    rows = sources_by_id()
    return {source_id: rows.get(source_id) for source_id in wanted}


def submit_backfill(documents, repeat):
    """Register a real corpus and start the throughput clock after registration."""
    source_ids = []
    accepted_ms = []
    for copy in range(repeat):
        for document in documents:
            doc = dict(document)
            if repeat > 1:
                doc["title"] = f"{doc.get('title') or pathlib.Path(doc['uri']).stem} benchmark-{copy + 1}"
            source_id, ms = request_document(doc)
            source_ids.append(source_id)
            accepted_ms.append(ms)
    return source_ids, accepted_ms, time.monotonic()


def chunks_per_second(rows, started):
    elapsed = max(time.monotonic() - started, 0.001)
    chunks = sum(int((row or {}).get("chunk_count") or 0) for row in rows.values()
                 if (row or {}).get("status") == "indexed")
    return chunks / elapsed, chunks, elapsed


def error_rate_pct(rows):
    if not rows:
        return 100.0
    failed = sum((row or {}).get("status") != "indexed" for row in rows.values())
    return 100.0 * failed / len(rows)


def expected_targets(row):
    expected = row.get("expected") or row.get("expect") or []
    if isinstance(expected, dict):
        expected = [expected]
    if not expected and row.get("sourceId"):
        expected = [{"sourceId": row["sourceId"], "kind": row.get("kind"),
                     "locator": row.get("locator", {})}]
    if not expected:
        raise RuntimeError(f"Query fixture has no expected citation: {row}")
    return expected


def citation_matches(citation, target):
    source_id = target.get("sourceId") or target.get("source_id") or target.get("id")
    if source_id and (citation.get("sourceId") or citation.get("source_id")
                      or citation.get("video_id")) != source_id:
        return False
    if target.get("title") and citation.get("title") != target["title"]:
        return False
    if target.get("kind") and citation.get("kind") != target["kind"]:
        return False
    locator = target.get("locator") or {}
    actual = citation.get("locator") or {}
    return all(actual.get(key) == value for key, value in locator.items())


def measure_recall_at_10(queries):
    hits = 0
    for row in queries:
        question = row.get("query") or row.get("question")
        if not question:
            raise RuntimeError(f"Query fixture has no query: {row}")
        st, text, _ = _req("POST", "/api/ask", body={"question": question, "top_k": 10})
        if st != 200:
            continue
        citations = _json(text, "POST /api/ask").get("citations", [])[:10]
        if any(citation_matches(citation, target)
               for target in expected_targets(row) for citation in citations):
            hits += 1
    return hits / len(queries) if queries else 0.0, hits, len(queries)


def run_command(name, command):
    if not command:
        raise RuntimeError(f"{name} is required for --resilience")
    try:
        subprocess.run(shlex.split(command), check=True, timeout=60)
    except (subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"{name} failed: {exc}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resilience", action="store_true")
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--documents", default=str(ROOT / "benchmark" / "documents.jsonl"))
    ap.add_argument("--queries", default=str(ROOT / "benchmark" / "queries.jsonl"))
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat corpus for concurrent load (distinct source files only)")
    ap.add_argument("--timeout", type=float, default=float(os.getenv("BENCH_TIMEOUT_S", "1800")))
    ap.add_argument("--poll", type=float, default=float(os.getenv("BENCH_POLL_S", "2")))
    ap.add_argument("--search-samples", type=int, default=40)
    ap.add_argument("--accept-samples", type=int, default=30)
    args = ap.parse_args()

    results, failures = {}, []

    def gate(name, value, ok, target):
        results[name] = {"value": value, "target": target, "pass": bool(ok)}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value} (target {target})")
        if not ok:
            failures.append(name)

    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    if args.repeat > 1:
        ap.error("--repeat cannot exceed 1: source-hash dedup makes repeated documents skip. "
                 "Add distinct documents to the fixture to create a larger load.")
    try:
        documents = load_jsonl(args.documents)
        queries = [] if args.resilience else load_jsonl(args.queries)
        if args.resilience:
            source_ids, _, started = submit_backfill(documents, args.repeat)
            # Wait until a worker has claimed work so the kill actually probes
            # in-flight recovery rather than an idle process.
            deadline = time.monotonic() + min(args.timeout, 120)
            while time.monotonic() < deadline:
                active = sources_by_id()
                if any((active.get(source_id) or {}).get("status") not in {"pending", "queued"}
                       for source_id in source_ids):
                    break
                time.sleep(args.poll)
            run_command("BENCH_KILL_WORKER_CMD", os.getenv("BENCH_KILL_WORKER_CMD", ""))
            run_command("BENCH_RESTART_WORKER_CMD", os.getenv("BENCH_RESTART_WORKER_CMD", ""))
            rows = wait_for_terminal(source_ids, args.timeout, args.poll)
            indexed = all((row or {}).get("status") == "indexed" for row in rows.values())
            rate, chunks, elapsed = chunks_per_second(rows, started)
            results["resilience_detail"] = {"sources": len(source_ids), "chunks": chunks,
                                            "elapsed_s": round(elapsed, 2),
                                            "chunks_per_s": round(rate, 3)}
            gate("no_loss_under_crash", indexed, indexed and SLA["no_loss_required"],
                 "0 dropped, all indexed")
            if args.json_out:
                pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
                print(f"wrote {args.json_out}")
            return sys.exit(1 if failures else 0)

        # 1. accept latency
        a = measure_accept_latency(args.accept_samples)
        gate("accept_latency_p95_ms", round(a, 1),
             a <= SLA["accept_latency_p95_ms"], SLA["accept_latency_p95_ms"])

        # 2/4. Submit a real concurrent backfill, measure search while it runs,
        # then use the same completed backfill for throughput.
        idle = measure_search_p95(args.search_samples)
        source_ids, _, started = submit_backfill(documents, args.repeat)
        during = measure_search_p95(args.search_samples)
        rows = wait_for_terminal(source_ids, args.timeout, args.poll)
        ratio = (during / idle) if idle else float("inf")
        gate("search_p95_during_ingest_ratio", round(ratio, 2),
             ratio <= SLA["search_p95_during_ingest_ratio_max"],
             SLA["search_p95_during_ingest_ratio_max"])
        errors = error_rate_pct(rows)
        gate("ingest_error_rate_pct", round(errors, 2),
             errors <= SLA["error_rate_max_pct"], SLA["error_rate_max_pct"])

        # 3. recall@10 on labeled queries.
        recall, hit_count, query_count = measure_recall_at_10(queries)
        results["recall_detail"] = {"hits": hit_count, "queries": query_count}
        gate("recall_at_10", round(recall, 3), recall >= SLA["recall_at_10_min"],
             SLA["recall_at_10_min"])

        throughput, chunks, elapsed = chunks_per_second(rows, started)
        results["throughput_detail"] = {"chunks": chunks, "elapsed_s": round(elapsed, 2)}
        gate("ingest_throughput_chunks_per_s", round(throughput, 3),
             throughput >= SLA["ingest_throughput_min_chunks_per_s"],
             SLA["ingest_throughput_min_chunks_per_s"])
    except RuntimeError as exc:
        print(f"BENCHMARK SETUP ERROR: {exc}", file=sys.stderr)
        return sys.exit(2)

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")

    print(f"\n{'ALL SLAs PASS' if not failures else 'SLA FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
