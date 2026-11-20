"""Deterministic metrics and aggregation.

Nothing in this module calls a model, so every function here is cheap,
repeatable and unit-tested. LLM-judged metrics live in judge.py.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable

NCT_RE = re.compile(r"\bNCT\d{8}\b")

# Phrases that signal the system declined to answer. Used together with the
# judge's own abstention verdict, never on its own.
ABSTAIN_MARKERS = (
    "no relevant",
    "not found in the",
    "no trials",
    "cannot answer",
    "can't answer",
    "unable to find",
    "insufficient information",
    "not enough information",
    "do not contain",
    "does not contain",
    "don't contain",
    "no information",
)

TARGETS = {
    # metric: (target, higher_is_better)
    "faithfulness": (0.90, True),
    "hallucination_rate": (0.05, False),
    "citation_precision": (0.95, True),
    "citation_support": (0.90, True),
    "answer_relevancy": (0.85, True),
    "abstention_accuracy": (0.90, True),
    "retrieval_recall": (0.80, True),
    "latency_p95_s": (10.0, False),
}


def extract_nct_ids(text: str) -> list[str]:
    """NCT IDs cited in the answer text, in order, de-duplicated."""
    seen: dict[str, None] = {}
    for m in NCT_RE.findall(text or ""):
        seen.setdefault(m, None)
    return list(seen)


def citation_precision(cited: Iterable[str], context_ids: Iterable[str]) -> float | None:
    """Share of cited NCT IDs that were actually in the context the model saw.

    A cited ID that is not in the context is a fabricated citation.
    Returns None when the answer cites nothing (not applicable, not zero).
    """
    cited = list(cited)
    if not cited:
        return None
    ctx = set(context_ids)
    return sum(c in ctx for c in cited) / len(cited)


def retrieval_recall(retrieved: Iterable[str], expected: Iterable[str]) -> float | None:
    """Share of expected NCT IDs that the retriever returned. None if unlabelled."""
    expected = set(expected)
    if not expected:
        return None
    return len(expected & set(retrieved)) / len(expected)


def looks_like_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in ABSTAIN_MARKERS)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _base(records: list[dict]) -> dict:
    """Run-level metrics without the per-category breakdown.

    Each metric is averaged only over the questions it applies to, and the
    count used is reported next to it (n_*), so a score is never inflated by
    questions where it was not measurable.
    """
    ok = [r for r in records if not r.get("error")]
    answerable = [r for r in ok if not r["expect"].get("should_abstain")]
    unanswerable = [r for r in ok if r["expect"].get("should_abstain")]

    def vals(rows, key):
        return [r["metrics"].get(key) for r in rows]

    def mean_n(rows, key):
        v = vals(rows, key)
        return _mean(v), sum(x is not None for x in v)

    halluc = [r["metrics"]["hallucinated"] for r in ok
              if r["metrics"].get("hallucinated") is not None]
    abst = [r["metrics"]["abstained_correctly"] for r in unanswerable
            if r["metrics"].get("abstained_correctly") is not None]
    lat = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]

    out = {"n_questions": len(records), "n_errors": len(records) - len(ok),
           "n_unanswerable": len(unanswerable)}
    for key in ("faithfulness", "citation_precision", "citation_support",
                "answer_relevancy"):
        out[key], out[f"n_{key}"] = mean_n(answerable, key)
    out["retrieval_recall"], out["n_retrieval_recall"] = mean_n(ok, "retrieval_recall")
    out["hallucination_rate"] = (sum(halluc) / len(halluc)) if halluc else None
    out["n_hallucination_rate"] = len(halluc)
    out["abstention_accuracy"] = (sum(abst) / len(abst)) if abst else None
    out["n_abstention_accuracy"] = len(abst)
    out["latency_p50_s"] = percentile(lat, 0.50)
    out["latency_p95_s"] = percentile(lat, 0.95)
    out["cost_per_query_usd"] = statistics.fmean(costs) if costs else None
    out["total_cost_usd"] = sum(costs) if costs else None
    out["n_costed"] = len(costs)
    return out


def summarize(records: list[dict]) -> dict:
    """Run-level metrics plus the same metrics broken down by category."""
    out = _base(records)
    out["by_category"] = {
        cat: _base([r for r in records if r["category"] == cat])
        for cat in sorted({r["category"] for r in records})
    }
    return out


def meets_target(metric: str, value: float | None) -> bool | None:
    if value is None or metric not in TARGETS:
        return None
    target, higher = TARGETS[metric]
    return value >= target if higher else value <= target
