"""Run the eval set against a system and write results for the dashboard.

    python -m evals.run_eval --system grok --label baseline
    python -m evals.run_eval --system grok --limit 5          # quick smoke test
    streamlit run evals/dashboard.py

Needs GROK_API_KEY (system under test) and ANTHROPIC_API_KEY (judge).
The first run downloads trials and FAERS data and freezes them under
evals/data/ so every later run is scored against identical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import metrics  # noqa: E402
from evals.judge import ClaudeJudge, scores_from_verdict  # noqa: E402
from evals.pricing import calls_cost  # noqa: E402

EVAL_DIR = ROOT / "evals"
DATA_DIR = EVAL_DIR / "data"
TRIALS_SNAPSHOT = DATA_DIR / "trials_snapshot.json"
FAERS_SNAPSHOT = DATA_DIR / "faers_snapshot.json"
CONDITIONS = ["lung cancer", "non-small cell lung cancer", "EGFR mutation",
              "KRAS mutation cancer"]  # same as app.py


def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [q["id"] for q in qs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate question ids: {sorted(dupes)}")
    return qs


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def build_corpus(refresh: bool = False):
    """Chroma collection built from the frozen trial snapshot."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from src.etl import fetch_clinical_trials, ingest_trials

    if refresh or not TRIALS_SNAPSHOT.exists():
        trials = []
        for cond in CONDITIONS:
            trials.extend(fetch_clinical_trials(condition=cond, max_studies=40))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TRIALS_SNAPSHOT.write_text(json.dumps(trials, indent=1))
        print(f"froze {len(trials)} trials -> {TRIALS_SNAPSHOT}")
    trials = json.loads(TRIALS_SNAPSHOT.read_text())
    # The same trial often comes back under several conditions; Chroma rejects
    # duplicate ids inside one upsert, so keep the first copy of each.
    unique = {}
    for t in trials:
        unique.setdefault(t["nct_id"], t)
    trials = list(unique.values())

    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    collection = chromadb.Client().get_or_create_collection("eval_trials")
    ingest_trials(trials, collection, embed_model)
    return collection, embed_model, trials


def make_system(name: str, collection, embed_model):
    from evals.systems import ClaudeAgentSystem, GrokPipelineSystem
    if name == "grok":
        return GrokPipelineSystem(collection, embed_model, faers_snapshot=FAERS_SNAPSHOT)
    if name == "claude":
        return ClaudeAgentSystem(collection, embed_model)
    raise ValueError(name)


def evaluate_one(q: dict, system, judge) -> dict:
    expect = q.get("expect", {})
    rec = {"id": q["id"], "category": q["category"], "question": q["question"],
           "drug": q.get("drug", ""), "expect": expect, "metrics": {}, "error": None}
    try:
        out = system.answer(q["question"], q.get("drug", ""))
    except Exception as e:
        rec["error"] = f"system: {type(e).__name__}: {e}"
        return rec

    cited = metrics.extract_nct_ids(out.answer)
    rec.update({
        "answer": out.answer,
        "context_text": out.context_text,
        "context_nct_ids": out.context_nct_ids,
        "retrieved_nct_ids": out.retrieved_nct_ids,
        "cited_nct_ids": cited,
        "calls": out.calls,
        "latency_s": out.latency_s,
        "cost_usd": calls_cost(out.calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in out.calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in out.calls),
        "extra": out.extra,
    })
    should_abstain = bool(expect.get("should_abstain"))
    rec["metrics"]["citation_precision"] = (
        None if should_abstain else metrics.citation_precision(cited, out.context_nct_ids))
    rec["metrics"]["fabricated_citations"] = [c for c in cited if c not in out.context_nct_ids]
    rec["metrics"]["retrieval_recall"] = metrics.retrieval_recall(
        out.retrieved_nct_ids, expect.get("expected_nct_ids", []))

    if judge is None:
        return rec
    try:
        j = judge.judge(q["question"], out.answer, out.context_text)
    except Exception as e:
        rec["error"] = f"judge: {e}"
        return rec
    rec["judge"] = j["verdict"]
    rec["judge_usage"] = j["usage"]
    rec["metrics"].update(scores_from_verdict(j["verdict"], should_abstain))
    return rec


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["grok", "claude"], default="grok")
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judging (deterministic metrics only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--category", default=None, help="only run one category")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--label", default="", help="short tag shown on the dashboard, e.g. baseline")
    ap.add_argument("--refresh-snapshot", action="store_true", help="re-download trial data (breaks comparability)")
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR / "results")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    collection, embed_model, trials = build_corpus(args.refresh_snapshot)
    system = make_system(args.system, collection, embed_model)
    judge = None if args.no_judge else ClaudeJudge(model=args.judge_model)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.system}" + (f"_{args.label}" if args.label else "")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run {run_id}: {len(questions)} questions, {len(trials)} trials in corpus")
    records, t0 = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = {pool.submit(evaluate_one, q, system, judge): q for q in questions}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            records.append(r)
            flag = "ERR " if r["error"] else ("HALL" if r["metrics"].get("hallucinated") else "ok  ")
            print(f"[{i:>3}/{len(questions)}] {flag} {r['id']}  {r.get('latency_s', 0):.1f}s")
    records.sort(key=lambda r: r["id"])

    with (run_dir / "records.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary = metrics.summarize(records)
    judge_costs = [r["judge_usage"]["cost_usd"] for r in records
                   if r.get("judge_usage", {}).get("cost_usd") is not None]
    config = {
        "run_id": run_id, "label": args.label, "system": system.name,
        "judge_model": None if judge is None else args.judge_model,
        "questions_file": args.questions.name, "questions_hash": file_hash(args.questions),
        "trials_snapshot_hash": file_hash(TRIALS_SNAPSHOT),
        "faers_snapshot_hash": file_hash(FAERS_SNAPSHOT),
        "n_trials": len(trials), "git_sha": git_sha(),
        "started_utc": stamp, "wall_time_s": time.perf_counter() - t0,
        "judge_cost_usd": sum(judge_costs) if judge_costs else None,
        "is_sample": False,
    }
    (run_dir / "summary.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2))

    print(f"\nwrote {run_dir}")
    for k in ("faithfulness", "hallucination_rate", "citation_precision", "citation_support",
              "answer_relevancy", "abstention_accuracy", "latency_p95_s", "cost_per_query_usd"):
        v = summary.get(k)
        print(f"  {k:<22} {'n/a' if v is None else round(v, 3)}")


if __name__ == "__main__":
    main()
