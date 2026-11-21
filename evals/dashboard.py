"""Eval dashboard:  streamlit run evals/dashboard.py

Reads every run under evals/results/ (and evals/sample_results/ when there
are no real runs yet) and lets you compare up to three runs side by side.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

EVAL_DIR = Path(__file__).resolve().parent
RESULT_DIRS = [EVAL_DIR / "results", EVAL_DIR / "sample_results"]

import sys  # noqa: E402
sys.path.insert(0, str(EVAL_DIR.parent))
from evals.metrics import TARGETS, meets_target  # noqa: E402

# Validated categorical slots (dataviz reference palette, first three:
# the only count that clears every check all-pairs). Color follows the run's
# position in the selection, so a run keeps its color while others come and go.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"],
          "dark": ["#3987e5", "#d95926", "#199e70"]}
MAX_RUNS = 3

RATE_METRICS = [  # (key, label, higher_is_better)
    ("faithfulness", "Faithfulness", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("citation_precision", "Citation precision", True),
    ("citation_support", "Citation support", True),
    ("answer_relevancy", "Answer relevancy", True),
    ("abstention_accuracy", "Abstention accuracy", True),
    ("retrieval_recall", "Retrieval recall", True),
]
HELP = {
    "faithfulness": "Share of answer claims supported by the context the model saw.",
    "hallucination_rate": "Share of questions whose answer has at least one unsupported or contradicted claim.",
    "citation_precision": "Share of cited NCT IDs that were actually in the context. The rest are fabricated.",
    "citation_support": "Of claims carrying an NCT citation, share where that trial supports the claim.",
    "answer_relevancy": "Judge's 1-5 relevancy score, rescaled to 0-1. Answerable questions only.",
    "abstention_accuracy": "Unanswerable questions where the system declined without inventing facts.",
    "retrieval_recall": "Expected NCT IDs that the retriever returned. Needs expected_nct_ids labels.",
}


# ---------------------------------------------------------------- data ----

@st.cache_data(ttl=30)
def load_runs() -> dict[str, dict]:
    runs = {}
    real = [p for p in (EVAL_DIR / "results").glob("*/summary.json")] if (EVAL_DIR / "results").exists() else []
    dirs = [EVAL_DIR / "results"] if real else RESULT_DIRS
    for d in dirs:
        if not d.exists():
            continue
        for s in sorted(d.glob("*/summary.json")):
            data = json.loads(s.read_text())
            recs = [json.loads(line) for line in (s.parent / "records.jsonl").read_text().splitlines()
                    if line.strip()]
            runs[data["config"]["run_id"]] = {**data, "records": recs}
    return runs


def fmt(metric: str, v) -> str:
    if v is None:
        return "n/a"
    if metric.startswith("latency"):
        return f"{v:.1f}s"
    if "cost" in metric:
        return f"${v:.4f}"
    return f"{v:.0%}" if v <= 1 else f"{v:.2f}"


def status_label(metric: str, v) -> str:
    ok = meets_target(metric, v)
    if ok is None:
        return "—"
    target, higher = TARGETS[metric]
    t = fmt(metric, target)
    return f"✅ meets target ({'≥' if higher else '≤'} {t})" if ok else f"⚠️ misses target ({'≥' if higher else '≤'} {t})"


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


# ------------------------------------------------------------------ ui ----

st.set_page_config(page_title="ClinicalAI Insights · Evals", page_icon="📊", layout="wide")
st.title("ClinicalAI Insights · Evaluation dashboard")

runs = load_runs()
if not runs:
    st.info("No runs yet. Run `python -m evals.run_eval --system grok --label baseline`, then reload.")
    st.stop()

run_ids = sorted(runs, reverse=True)
label_of = {r: (f"{runs[r]['config'].get('label') or r}  ·  {runs[r]['config']['system']}  ·  {r[:15]}")
            for r in run_ids}
# Shorter names for chart legends and axes; the full label is in the picker.
short_of = {r: (runs[r]['config'].get('label') or r[:15]) for r in run_ids}

selected = st.multiselect(
    f"Runs to compare (up to {MAX_RUNS}; the first is the baseline)",
    run_ids, default=run_ids[:1], max_selections=MAX_RUNS, format_func=lambda r: label_of[r])
if not selected:
    st.stop()

if any(runs[r]["config"].get("is_sample") for r in selected):
    st.warning("**Sample data.** These runs were generated to demo the dashboard, not by running "
               "the system. Run the harness to replace them with real numbers.", icon="🧪")

configs = [runs[r]["config"] for r in selected]
if len({(c["questions_hash"], c["trials_snapshot_hash"]) for c in configs}) > 1:
    st.error("These runs used different question sets or trial snapshots, so their scores "
             "are not directly comparable.", icon="⚠️")

palette = SERIES[theme_mode()]
color_of = {short_of[r]: palette[i] for i, r in enumerate(selected)}
color_scale = alt.Scale(domain=list(color_of), range=list(color_of.values()))
base_id = selected[0]
base = runs[base_id]["summary"]

tab_over, tab_cat, tab_perf, tab_q = st.tabs(["Overview", "By category", "Latency & cost", "Questions"])

# ------------------------------------------------------------ overview ----
with tab_over:
    c = runs[base_id]["config"]
    st.caption(f"Baseline: **{label_of[base_id]}** · judge `{c.get('judge_model') or 'none'}` · "
               f"{base['n_questions']} questions ({base['n_errors']} errors) · {c['n_trials']} trials · "
               f"commit `{c.get('git_sha') or '?'}`")

    tiles = [m for m in RATE_METRICS if m[0] != "retrieval_recall" or base.get("retrieval_recall") is not None]
    tiles += [("latency_p95_s", "Latency p95", False), ("cost_per_query_usd", "Cost / query", False)]
    cols = st.columns(len(tiles) // 2 + len(tiles) % 2)
    for i, (key, label, higher) in enumerate(tiles):
        v = base.get(key)
        delta = None
        if len(selected) > 1:
            other = runs[selected[1]]["summary"].get(key)
            if v is not None and other is not None:
                d = other - v
                delta = (f"{d * 100:+.0f} pts" if not key.startswith(("latency", "cost")) else
                         (f"{d:+.1f}s" if key.startswith("latency") else f"${d:+.4f}"))
        with cols[i % len(cols)]:
            n = base.get(f"n_{key}")
            st.metric(label, fmt(key, v), delta=delta,
                      delta_color="normal" if higher else "inverse",
                      help=HELP.get(key, "") + (f" (n={n})" if n is not None else ""))
            st.caption(status_label(key, v))
    if len(selected) > 1:
        st.caption(f"Deltas show **{short_of[selected[1]]}** minus the baseline (rates in percentage points).")

    st.subheader("Quality metrics")
    rows = []
    for r in selected:
        s = runs[r]["summary"]
        for key, label, _ in RATE_METRICS:
            if s.get(key) is not None:
                rows.append({"run": short_of[r], "metric": label, "value": s[key],
                             "shown": fmt(key, s[key]), "n": s.get(f"n_{key}")})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No judged metrics in these runs (were they run with --no-judge?).")
    else:
        order = [lbl for _, lbl, _ in RATE_METRICS]
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.8}).encode(
            y=alt.Y("metric:N", sort=order, title=None),
            yOffset=alt.YOffset("run:N", sort=[short_of[r] for r in selected]),
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=True, tickCount=5), title=None),
            color=alt.Color("run:N", scale=color_scale, legend=alt.Legend(orient="top", title=None, labelLimit=400)),
            tooltip=[alt.Tooltip("run:N"), alt.Tooltip("metric:N"), alt.Tooltip("shown:N", title="value"),
                     alt.Tooltip("n:Q", title="questions")],
        )
        labels = bars.mark_text(align="left", dx=4, color="gray").encode(text="shown:N", color=alt.value("gray"))
        st.altair_chart((bars + labels).properties(height=60 * len(order) * max(1, len(selected)) / 2 + 60),
                        use_container_width=True)
        st.caption("Hallucination rate is lower-is-better; every other bar is higher-is-better.")
        with st.expander("Table view"):
            st.dataframe(df.pivot(index="metric", columns="run", values="shown").reindex(order),
                         use_container_width=True)

    # Worst offenders: the questions an interviewer will ask you about.
    st.subheader("Hallucinations and fabricated citations (baseline)")
    bad = [r for r in runs[base_id]["records"]
           if r["metrics"].get("hallucinated") or r["metrics"].get("fabricated_citations")]
    if not bad:
        st.success("None in this run.", icon="✅")
    for r in bad:
        fab = r["metrics"].get("fabricated_citations") or []
        st.markdown(f"- **{r['id']}** ({r['category']}): {r['question']}"
                    + (f"  \n  {r['metrics'].get('n_unsupported', 0)} unsupported claim(s)"
                       if r["metrics"].get("hallucinated") else "")
                    + (f"  \n  fabricated citations: `{', '.join(fab)}`" if fab else ""))

# ------------------------------------------------------------ category ----
with tab_cat:
    metric_key = st.selectbox("Metric", [k for k, _, _ in RATE_METRICS],
                              format_func=lambda k: dict((a, b) for a, b, _ in RATE_METRICS)[k])
    rows = []
    for r in selected:
        for cat, s in runs[r]["summary"]["by_category"].items():
            if s.get(metric_key) is not None:
                rows.append({"run": short_of[r], "category": cat, "value": s[metric_key],
                             "shown": fmt(metric_key, s[metric_key]), "n": s.get(f"n_{metric_key}")})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("This metric is not measured for any category in these runs.")
    else:
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("category:N", title=None),
            yOffset=alt.YOffset("run:N", sort=[short_of[r] for r in selected]),
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5), title=None),
            color=alt.Color("run:N", scale=color_scale, legend=alt.Legend(orient="top", title=None, labelLimit=400)),
            tooltip=["run:N", "category:N", alt.Tooltip("shown:N", title="value"),
                     alt.Tooltip("n:Q", title="questions")],
        )
        labels = bars.mark_text(align="left", dx=4).encode(text="shown:N", color=alt.value("gray"))
        st.altair_chart((bars + labels).properties(height=90 * df["category"].nunique()),
                        use_container_width=True)
        st.caption(HELP[metric_key])
        with st.expander("Table view"):
            st.dataframe(df.pivot(index="category", columns="run", values="shown"), use_container_width=True)

# ---------------------------------------------------------------- perf ----
with tab_perf:
    rows = []
    for r in selected:
        for rec in runs[r]["records"]:
            if rec.get("latency_s") is not None:
                rows.append({"run": short_of[r], "id": rec["id"], "category": rec["category"],
                             "latency_s": rec["latency_s"], "cost_usd": rec.get("cost_usd"),
                             "tokens": rec.get("input_tokens", 0) + rec.get("output_tokens", 0)})
    df = pd.DataFrame(rows)
    if not df.empty:
        st.subheader("Latency per question")
        dots = alt.Chart(df).mark_circle(size=80, opacity=0.8, stroke="white", strokeWidth=2).encode(
            x=alt.X("latency_s:Q", title="seconds"),
            y=alt.Y("run:N", title=None, axis=alt.Axis(labelLimit=300), sort=[short_of[r] for r in selected]),
            yOffset=alt.YOffset("jitter:Q"),
            color=alt.Color("run:N", scale=color_scale, legend=None),
            tooltip=["run:N", "id:N", "category:N", alt.Tooltip("latency_s:Q", format=".2f", title="seconds")],
        ).transform_calculate(jitter="random()")
        rule = alt.Chart(pd.DataFrame({"x": [TARGETS["latency_p95_s"][0]]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(x="x:Q")
        st.altair_chart((dots + rule).properties(height=110 * len(selected)), use_container_width=True)
        st.caption(f"Dashed line: {TARGETS['latency_p95_s'][0]:.0f}s p95 target.")

    table = []
    for r in selected:
        s, c = runs[r]["summary"], runs[r]["config"]
        table.append({"run": label_of[r], "p50 latency": fmt("latency", s.get("latency_p50_s")),
                      "p95 latency": fmt("latency", s.get("latency_p95_s")),
                      "cost / query": fmt("cost", s.get("cost_per_query_usd")),
                      "total system cost": fmt("cost", s.get("total_cost_usd")),
                      "judge cost": fmt("cost", c.get("judge_cost_usd")),
                      "priced questions": f"{s.get('n_costed', 0)}/{s['n_questions']}"})
    st.dataframe(pd.DataFrame(table).set_index("run"), use_container_width=True)
    if any(runs[r]["summary"].get("cost_per_query_usd") is None for r in selected):
        st.caption("Cost shows n/a when a model has no price in evals/pricing.py (Grok is left blank on purpose).")

# ----------------------------------------------------------- questions ----
with tab_q:
    run_q = st.selectbox("Run", selected, format_func=lambda r: label_of[r], key="qrun")
    recs = runs[run_q]["records"]
    c1, c2 = st.columns(2)
    cats = c1.multiselect("Category", sorted({r["category"] for r in recs}))
    only = c2.radio("Show", ["All", "Hallucinated", "Fabricated citation", "Errors"], horizontal=True)

    def keep(r):
        if cats and r["category"] not in cats:
            return False
        if only == "Hallucinated":
            return bool(r["metrics"].get("hallucinated"))
        if only == "Fabricated citation":
            return bool(r["metrics"].get("fabricated_citations"))
        if only == "Errors":
            return bool(r.get("error"))
        return True

    shown = [r for r in recs if keep(r)]
    st.caption(f"{len(shown)} of {len(recs)} questions")
    for r in shown:
        m = r["metrics"]
        flag = ("❌ error" if r.get("error") else "⚠️ hallucinated" if m.get("hallucinated")
                else "✅ grounded")
        with st.expander(f"{r['id']} · {r['category']} · {flag} · {r['question']}"):
            if r.get("error"):
                st.error(r["error"])
                continue
            a, b, cc, d = st.columns(4)
            a.metric("Faithfulness", fmt("faithfulness", m.get("faithfulness")))
            b.metric("Citation precision", fmt("citation_precision", m.get("citation_precision")))
            cc.metric("Latency", fmt("latency", r.get("latency_s")))
            d.metric("Cost", fmt("cost", r.get("cost_usd")))
            if r["expect"].get("note"):
                st.caption(f"Why this question: {r['expect']['note']}")
            st.markdown("**Answer**")
            st.write(r.get("answer", ""))
            if m.get("fabricated_citations"):
                st.error(f"Cited but not in context: {', '.join(m['fabricated_citations'])}")
            claims = (r.get("judge") or {}).get("claims", [])
            if claims:
                st.markdown("**Judge's claim-by-claim verdict**")
                icon = {"supported": "✅ supported", "unsupported": "⚠️ unsupported", "contradicted": "❌ contradicted"}
                st.dataframe(pd.DataFrame([{"claim": c["text"], "verdict": icon.get(c["label"], c["label"]),
                                            "cited": ", ".join(c.get("cited_nct_ids", [])),
                                            "reason": c.get("reason", "")} for c in claims]),
                             use_container_width=True, hide_index=True)
            with st.popover("Context the model saw"):
                st.code(r.get("context_text") or "(empty)", language=None)
