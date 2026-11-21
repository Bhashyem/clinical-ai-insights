# Evals

A repeatable evaluation harness and dashboard for ClinicalAI Insights.
It scores the existing Grok/LangGraph pipeline today, and the Claude rebuild
plugs into the same harness so the two can be compared on identical data.

## Quick start

```bash
pip install -r requirements.txt
export GROK_API_KEY=...        # system under test
export ANTHROPIC_API_KEY=...   # judge

python -m evals.run_eval --system grok --limit 5 --label smoke   # ~1 min sanity check
python -m evals.run_eval --system grok --label baseline           # full set
streamlit run evals/dashboard.py
```

Until a real run exists, the dashboard shows two **SAMPLE** runs (made by
`python -m evals.make_sample`) under a "Sample data" banner. Once a run
exists in `evals/results/`, the samples are hidden.

## What gets measured

| Metric | How | Applies to |
|---|---|---|
| Faithfulness | Judge splits the answer into atomic claims and labels each against the **exact context the model saw** | Answerable questions |
| Hallucination rate | Share of answers with at least one unsupported or contradicted claim | All questions |
| Citation precision | Cited NCT IDs that were really in the context (deterministic, no LLM) | Answerable questions |
| Citation support | Of claims carrying a citation, share where that trial supports the claim | Answerable questions |
| Answer relevancy | Judge's 1–5 score rescaled to 0–1 | Answerable questions |
| Abstention accuracy | Declined **and** invented nothing | Unanswerable questions |
| Retrieval recall | Expected NCT IDs the retriever returned | Questions you have labelled |
| Latency p50 / p95 | Wall clock per question | All questions |
| Cost per query | Token usage × `evals/pricing.py` | Priced models only |

Each run-level number is averaged only over the questions where it applies,
and the dashboard shows that count (n) next to it.

## Design choices worth explaining in an interview

- **Judge against what the model saw, not what was retrieved.** The pipeline
  retrieves 5 trials but the synthesis prompt only includes the top 3, cut to
  200 characters. The adapter captures the real prompt, so a claim drawn from
  trial #5 counts as unsupported, and citing trial #5 counts as a fabricated
  citation. The notebook's earlier eval judged against all 5 trials at 300
  characters, which over-credited the system.
- **A different model judges.** The notebook used grok-3-mini to grade
  grok-3-mini. Here Claude judges, using a forced tool call so verdicts are
  structured, and at temperature 0. When you evaluate the Claude rebuild,
  re-run with a second judge and report both.
- **Scores are computed in code from labels.** The judge only labels claims;
  Python does the arithmetic. Every number can be traced back to
  claim-level verdicts on the Questions tab.
- **Frozen data.** The first run saves trials to `evals/data/trials_snapshot.json`
  and FAERS responses to `evals/data/faers_snapshot.json`. Later runs reuse
  them, so a score change means the system changed, not the data. The
  dashboard warns if you compare runs built on different snapshots or
  question sets. `--refresh-snapshot` re-downloads on purpose.
- **Unanswerable and false-premise questions.** A RAG system that answers
  everything looks great on faithfulness and still fails in production.
  Ten questions have no answer in the data, and six carry a false premise or a
  prompt injection.
- **Unknown cost shows as n/a, not $0.** Grok prices are blank in
  `evals/pricing.py` until you fill them in from x.ai.

## The question set

`questions.jsonl`, one JSON object per line:

```json
{"id": "tria-001", "category": "trials", "question": "...", "drug": "",
 "expect": {"should_abstain": false, "expected_nct_ids": []}}
```

54 questions: 20 trial lookups, 10 FAERS safety, 8 hybrid, 10 unanswerable,
6 false-premise. To grow it toward 100–200:

1. Add questions in the same shape. `drug` is required for safety and hybrid.
2. To turn on retrieval recall, fill `expected_nct_ids` with the trials that
   should come back. The easy way: after a run, open the Questions tab, check
   which retrieved trials are genuinely relevant, and copy their IDs in.
3. `python -m pytest tests -q` checks the file is well formed.

## Plugging in the Claude rebuild

Implement `ClaudeAgentSystem.answer()` in `evals/systems.py` (the docstring
lists what to return), then:

```bash
python -m evals.run_eval --system claude --label claude-v1
python -m evals.run_eval --system claude --label claude-v1-judge2 --judge-model <another model>
```

Select both runs on the dashboard to compare them with the Grok baseline.

## Tests

```bash
python -m pytest tests -q
```

Offline: no network and no API keys. The tests cover the metrics, the judge
call shape and pricing, the Grok adapter's capture of context and usage
(with a fake client), fabricated-citation detection, error handling, the
question-set format, and a headless render of the dashboard.
