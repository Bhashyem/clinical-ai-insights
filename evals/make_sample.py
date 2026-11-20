"""Generate clearly-labelled SAMPLE runs so the dashboard can be previewed
before any real run exists. Uses fake systems and a fake judge, but pushes
them through the real evaluate_one() and summarize() code paths.

    python -m evals.make_sample
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import metrics  # noqa: E402
from evals.run_eval import evaluate_one, file_hash, load_questions  # noqa: E402
from evals.systems import SystemOutput  # noqa: E402

OUT = ROOT / "evals" / "sample_results"


class FakeSystem:
    """Behaves like a RAG system with tunable failure rates."""

    def __init__(self, name, model, seed, p_halluc, p_fabricate, p_abstain_ok, lat, tokens):
        self.name, self.model = name, model
        self.rng = random.Random(seed)
        self.p_halluc, self.p_fab, self.p_abst = p_halluc, p_fabricate, p_abstain_ok
        self.lat, self.tokens = lat, tokens
        self.answers: dict[str, dict] = {}

    def answer(self, question, drug=""):
        rng = self.rng
        ids = [f"NCT0{rng.randint(1000000, 9999999)}" for _ in range(5)]
        ctx = "\n\n".join(f"Trial {i}: Sample trial title\nPhase: PHASE2\nExcerpt: sample excerpt"
                          for i in ids[:3])
        unanswerable = any(k in question.lower() for k in (
            "alzheimer", "insulin", "velnatinib", "median overall", "hazard ratio", "when will",
            "should my", "how many patients", "near boston", "list price"))
        plan = {"halluc": False, "fab": False, "abstain": False}
        if unanswerable and rng.random() < self.p_abst:
            answer = "The retrieved trials do not contain information to answer this question."
            plan["abstain"] = True
        else:
            cite = ids[0] if not (rng.random() < self.p_fab) else "NCT09999999"
            plan["fab"] = cite == "NCT09999999"
            plan["halluc"] = unanswerable or plan["fab"] or rng.random() < self.p_halluc
            answer = f"Trial {cite} is a Phase 2 study relevant to this question. See also {ids[1]}."
        lat = max(0.8, rng.gauss(*self.lat))
        tin = int(rng.gauss(*self.tokens))
        self.answers[question] = plan
        return SystemOutput(
            answer=answer, context_text=ctx, context_nct_ids=ids[:3], retrieved_nct_ids=ids,
            calls=[{"model": self.model, "input_tokens": tin, "output_tokens": 180,
                    "cache_write_tokens": 0, "cache_read_tokens": int(tin * 0.6) if "claude" in self.model else 0,
                    "latency_s": lat, "role": "synthesis"}],
            latency_s=lat,
        )


class FakeJudge:
    def __init__(self, system):
        self.system = system

    def judge(self, question, answer, context_text):
        plan = self.system.answers[question]
        if plan["abstain"]:
            claims = []
        else:
            claims = [{"text": "The cited trial is a Phase 2 study.", "label": "supported",
                       "cited_nct_ids": metrics.extract_nct_ids(answer)[:1],
                       "citation_supports_claim": not plan["fab"], "reason": "sample"},
                      {"text": "It is relevant to the question.",
                       "label": "unsupported" if plan["halluc"] else "supported",
                       "cited_nct_ids": [], "citation_supports_claim": None, "reason": "sample"}]
        rel = 2 if plan["halluc"] else 5
        return {"verdict": {"claims": claims, "abstained": plan["abstain"], "relevancy": rel},
                "usage": {"model": "claude-sonnet-5-5", "input_tokens": 900, "output_tokens": 250,
                          "cache_write_tokens": 0, "cache_read_tokens": 700, "latency_s": 2.0,
                          "cost_usd": 0.0045}}


def write_run(run_id, label, system, questions):
    judge = FakeJudge(system)
    records = sorted((evaluate_one(q, system, judge) for q in questions), key=lambda r: r["id"])
    d = OUT / run_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "records.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    config = {"run_id": run_id, "label": label, "system": system.name, "judge_model": "claude-sonnet-5-5",
              "questions_file": "questions.jsonl",
              "questions_hash": file_hash(ROOT / "evals" / "questions.jsonl"),
              "trials_snapshot_hash": "sample", "faers_snapshot_hash": "sample", "n_trials": 142,
              "git_sha": "sample", "started_utc": run_id[:15], "wall_time_s": 0,
              "judge_cost_usd": sum(r["judge_usage"]["cost_usd"] for r in records if r.get("judge_usage")),
              "is_sample": True}
    (d / "summary.json").write_text(json.dumps({"config": config, "summary": metrics.summarize(records)}, indent=2))
    return d


def main():
    qs = load_questions(ROOT / "evals" / "questions.jsonl")
    # The sample Grok run shows cost as n/a, because Grok prices are blank in pricing.py.
    grok = FakeSystem("grok-langgraph", "grok-3-mini", 7, p_halluc=0.15, p_fabricate=0.08,
                      p_abstain_ok=0.5, lat=(6.5, 2.0), tokens=(1400, 200))
    claude = FakeSystem("claude-agent-sdk", "claude-sonnet-5-5", 11, p_halluc=0.05, p_fabricate=0.0,
                        p_abstain_ok=0.9, lat=(5.0, 1.5), tokens=(2600, 300))
    print(write_run("20261120-000000_grok_SAMPLE-baseline", "SAMPLE baseline", grok, qs))
    print(write_run("20261120-000100_claude_SAMPLE-rebuild", "SAMPLE claude rebuild", claude, qs))


if __name__ == "__main__":
    main()
