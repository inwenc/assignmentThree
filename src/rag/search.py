"""Read path: question -> retrieve -> gate -> cited answer (or honest abstain).

Retrieval is milliseconds; the multimodal LLM call is seconds and dominates
cost. So the shape is a confidence funnel: fetch KNN_K candidates, collapse
temporal near-duplicates, trim to TOP_K, and — Gate 1 — if even the best
score is below CONFIDENCE_THRESHOLD, abstain WITHOUT calling the LLM. That
one free check kills most hallucination risk. Generated answers get their
[n] citations validated; invented references are stripped.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import config, db, llm, storage
from ..config import (BRANCH_TOP_K, CONFIDENCE_THRESHOLD, CROSS_MODAL_BOOST,
                      FUSION_WINDOW_S, RRF_K, TEXT_CONFIDENCE_THRESHOLD, TOP_K)
from . import vector_store
from .embeddings import embed_query, embed_text

ABSTAIN = ("I couldn't find that in your sources — nothing indexed looks "
           "related to the question.")
_DOCUMENT_KINDS = {"paper", "deck"}


def _seconds(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _source_id(hit: dict) -> str:
    return str(hit.get("source_id") or hit["video_id"])


def _document_locator(hit: dict) -> tuple[str, int] | None:
    kind = hit.get("kind")
    if kind == "paper" and hit.get("page") is not None:
        return ("page", int(hit["page"]))
    if kind == "deck" and hit.get("slide") is not None:
        return ("slide", int(hit["slide"]))
    return None


def _fuse(visual_hits: list[dict], text_hits: list[dict]) -> list[dict]:
    """Reciprocal-Rank-Fusion of the two branches into time windows.

    Raw scores are incomparable (CLIP ~0.3 vs bge ~0.7), so we rank each branch
    on its own and score by rank: rrf = 1/(RRF_K + rank). Then we bucket hits
    within FUSION_WINDOW_S seconds of each other (same video) into one 'moment',
    sum their rrf, and boost windows where BOTH modalities agree — two
    independent signals pointing at the same instant is the strongest evidence.
    """
    def ranked(hits, modality):
        out = []
        for rank, h in enumerate(hits):
            t = float(h.get("t_start", h.get("ms", 0) / 1000.0))
            out.append({**h, "modality": modality, "rrf": 1.0 / (RRF_K + rank), "t": t})
        return out

    windows: list[dict] = []
    # Hits arrive best-first (rrf desc), so the first hit landing in a window for
    # a given modality is that modality's best hit there.
    for h in sorted(ranked(visual_hits, "frame") + ranked(text_hits, "text"),
                    key=lambda x: x["rrf"], reverse=True):
        source_id = _source_id(h)
        locator = _document_locator(h)
        # Documents have no timeline. Merge only chunks that cite the same
        # exact page/slide; otherwise a paper's entire contents would collapse
        # into a fake t=0 moment.
        w = next((w for w in windows
                  if w["source_id"] == source_id
                  and ((locator is not None and w["locator"] == locator)
                       or (locator is None and w["locator"] is None
                           and abs(w["t"] - h["t"]) <= FUSION_WINDOW_S))), None)
        if w is None:
            w = {"source_id": source_id, "t": h["t"], "locator": locator,
                 "rrf": 0.0, "modalities": set(), "frame": None, "text": None}
            windows.append(w)
        w["modalities"].add(h["modality"])
        slot = "frame" if h["modality"] == "frame" else "text"
        # Keep only the BEST hit per modality. Summing every hit would let a
        # burst of near-identical frames clustered in one 15s window inflate its
        # score past a genuine frame+transcript match — the bug that ranked a
        # silent frame-burst above the moment that actually answered.
        if w[slot] is None:
            w[slot] = h
    for w in windows:
        # Score = best frame + best transcript hit; ×boost when BOTH modalities
        # agree at this instant (two independent signals = strongest evidence).
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0.0) + \
                   (w["text"]["rrf"] if w["text"] else 0.0)
        if {"frame", "text"} <= w["modalities"]:
            w["rrf"] *= CROSS_MODAL_BOOST
    windows.sort(key=lambda w: w["rrf"], reverse=True)
    return windows


def _video_deeplink(video: dict | None, video_id: str, ms: int) -> str:
    secs = ms // 1000
    if video and video.get("source") == "youtube" and video.get("url"):
        sep = "&" if "?" in video["url"] else "?"
        return f"{video['url']}{sep}t={secs}"
    return f"/api/video/{video_id}#t={secs}"


def _document_deeplink(source: dict | None, user_id: str, source_id: str,
                       kind: str, locator: dict[str, int]) -> str:
    """Open a source at its retrieved page/slide when the format supports it."""
    if source and source.get("url"):
        base = source["url"].split("#", 1)[0]
    elif source and source.get("storage_key") and storage.presign_capable():
        base = storage.presign_get(source["storage_key"])
    else:
        base = f"/api/document/{source_id}?u={user_id}"
    if kind == "paper":
        return f"{base}#page={locator['page']}"
    return f"{base}#slide={locator['slide']}"


def _thumb_url(user_id: str, video_id: str, idx: int) -> str:
    """Browser-facing thumbnail URL. Presigned GET straight to the bucket when
    the provider supports it (an <img> tag can't send auth headers); the API
    serves the bytes itself only in local-dev mode."""
    if storage.presign_capable():
        return storage.presign_get(storage.frame_key(user_id, video_id, idx))
    return f"/api/frame/{video_id}/{idx:06d}.jpg?u={user_id}"


def _media_url(video: dict | None, user_id: str, video_id: str) -> str | None:
    """Playback URL for uploaded videos (YouTube plays via its own URL)."""
    if not video or video.get("source") != "upload" or not video.get("storage_key"):
        return None
    if storage.presign_capable():
        return storage.presign_get(video["storage_key"])
    return f"/api/video/{video_id}?u={user_id}"


def retrieve(question: str, user_id: str, *, top_k: int | None = None,
             video_id: str | None = None,
             video_ids: list[str] | None = None) -> dict[str, Any]:
    """Multimodal retrieve: query BOTH branches (CLIP frames + transcript text),
    fuse by RRF into time windows, and return numbered moment-citations.

    Returns {citations, best_visual, best_text} — the two raw bests feed the
    confidence gate (RRF scores are too small to threshold on). video_ids scopes
    the search to chosen videos (UI select/unselect)."""
    k = top_k or TOP_K

    # Visual branch — CLIP text→image.
    vhits = vector_store.search(embed_text(question), user_id, top_k=BRANCH_TOP_K,
                                video_id=video_id, video_ids=video_ids)
    best_visual = vhits[0]["score"] if vhits else 0.0

    # Text branch — bge query→transcript/document chunks. It must run even when
    # video transcripts are disabled because papers and decks live here too.
    thits: list[dict] = []
    best_text = 0.0
    thits = vector_store.search_text(embed_query(question), user_id,
                                     top_k=BRANCH_TOP_K, video_id=video_id,
                                     video_ids=video_ids)
    best_text = thits[0]["score"] if thits else 0.0

    windows = _fuse(vhits, thits)[:k]
    sources = db.videos_by_ids(sorted({w["source_id"] for w in windows}))
    citations = []
    for i, w in enumerate(windows, 1):
        source_id = w["source_id"]
        meta = sources.get(source_id)
        fr, tx = w["frame"], w["text"]
        evidence = tx or fr or {}
        kind = evidence.get("kind") or (meta or {}).get("source")
        kind = kind if kind in _DOCUMENT_KINDS else "video"
        document_locator = _document_locator(evidence)
        # Anchor on the frame's exact timestamp when there is one (precise visual
        # seek); otherwise the transcript chunk's start.
        ms = int(fr["ms"]) if fr else int(w["t"] * 1000)
        idx = int(fr["idx"]) if fr else None
        if kind == "paper":
            locator = {"page": document_locator[1]}
            location = f"p. {locator['page']}"
            deeplink = _document_deeplink(meta, user_id, source_id, kind, locator)
        elif kind == "deck":
            locator = {"slide": document_locator[1]}
            location = f"slide {locator['slide']}"
            deeplink = _document_deeplink(meta, user_id, source_id, kind, locator)
        else:
            end_ms = int(float((tx or {}).get("t_end", ms / 1000.0)) * 1000)
            locator = {"start_ms": ms, "end_ms": end_ms}
            location = _seconds(ms)
            deeplink = _video_deeplink(meta, source_id, ms)
        citations.append({
            "n": i,
            "sourceId": source_id,
            "source_id": source_id,
            "video_id": source_id,  # UI/search compatibility during migration
            "kind": kind,
            "locator": locator,
            "title": (meta or {}).get("title") or source_id,
            "url": (meta or {}).get("url"),
            "source": (meta or {}).get("source"),
            "ms": ms,
            "timestamp": location,
            "idx": idx,
            "thumbnail": _thumb_url(user_id, source_id, idx) if idx is not None else None,
            "media_url": _media_url(meta, user_id, source_id),
            "deeplink": deeplink,
            "score": round(w["rrf"], 4),
            "transcript": (tx or {}).get("text"),
            "modalities": sorted(w["modalities"]),
        })
    return {"citations": citations, "best_visual": best_visual, "best_text": best_text}


def _fallback_answer(citations: list[dict[str, Any]]) -> str:
    """No-LLM summary: rank the visually-closest moments. Honest about being
    similarity, not synthesis."""
    top = citations[0]
    where = f"{top['title']} at {top['timestamp']}" if top.get("title") else top["timestamp"]
    others = ", ".join(f"{c['timestamp']} [{c['n']}]" for c in citations[1:4])
    msg = f"Closest retrieved source: {where} [{top['n']}] (similarity {top['score']})."
    if others:
        msg += f" Other relevant moments: {others}."
    return msg


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _validate_citations(answer: str, n_frames: int) -> str:
    """Strip invented [n] references the model has no frame for."""
    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [str(x) for x in nums if 1 <= x <= n_frames]
        return f"[{', '.join(valid)}]" if valid else ""
    return _CITE_RE.sub(fix, answer)


def _build_moments(user_id: str, citations: list[dict[str, Any]]) -> list[dict]:
    """Turn citations into what the LLM sees: each moment carries its frame
    image (if any) and/or its transcript excerpt (if any), numbered to match."""
    def frame_bytes(c):
        if c.get("idx") is None:
            return None
        try:
            return storage.get_bytes(storage.frame_key(user_id, c["video_id"], c["idx"]))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        images = list(ex.map(frame_bytes, citations))
    return [{"image": img, "transcript": c.get("transcript"),
             "timestamp": c["timestamp"]} for img, c in zip(images, citations)]


def resolve_llm(user_id: str) -> tuple[llm.LLMConfig | None, str]:
    """Which model answers for this tenant: their own hosted endpoint
    (ms_user_llms — e.g. a vLLM server) first, the server-wide LLM_* env
    config as fallback. Returns (config, source) with source in
    {"user", "server", "none"}."""
    row = db.get_user_llm(user_id)
    if row and row.get("model"):
        return llm.from_row(row), "user"
    cfg = llm.env_config()
    return (cfg, "server") if cfg else (None, "none")


def ask(question: str, user_id: str, *, top_k: int | None = None,
        video_id: str | None = None,
        video_ids: list[str] | None = None) -> dict[str, Any]:
    r = retrieve(question, user_id, top_k=top_k, video_id=video_id, video_ids=video_ids)
    citations = r["citations"]
    result: dict[str, Any] = {"question": question, "citations": citations}

    if not citations:
        result.update(answer="No relevant sources were found. Try ingesting a source first.",
                      llm_used=False, abstained=True)
        return result

    # Gate 1 — confidence on the RAW per-branch bests (not the RRF score).
    # Abstain only if NEITHER what's on screen nor what's said looks relevant.
    visual_ok = r["best_visual"] >= CONFIDENCE_THRESHOLD
    text_ok = r["best_text"] >= TEXT_CONFIDENCE_THRESHOLD
    if CONFIDENCE_THRESHOLD and not visual_ok and not text_ok:
        result.update(answer=ABSTAIN, llm_used=False, abstained=True)
        return result

    cfg, source = resolve_llm(user_id)
    if cfg is None:
        # No generative model — summarize the best matches instead of inventing.
        result.update(answer=_fallback_answer(citations), llm_used=False,
                      note=("Retrieval-only results. Connect your own model "
                            "(vLLM/Ollama/API) in settings, or set LLM_API_KEY "
                            "on the server, for a synthesized, grounded answer."))
        return result

    moments = _build_moments(user_id, citations)
    result["answer"] = _validate_citations(llm.answer(question, moments, cfg),
                                           len(citations))
    result["llm_used"] = True
    result["llm_source"] = source          # "user" = their own hosted model
    result["llm_model"] = cfg.model
    return result
