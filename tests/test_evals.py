"""Offline tests for the eval harness: no network, no API keys.

    python -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import metrics, pricing  # noqa: E402
from evals.judge import ClaudeJudge, scores_from_verdict  # noqa: E402
from evals.run_eval import evaluate_one, load_questions  # noqa: E402


# ---------------------------------------------------------------- metrics

def test_extract_nct_ids_dedupes_in_order():
    assert metrics.extract_nct_ids("See NCT01234567 and NCT07654321, again NCT01234567.") == [
        "NCT01234567", "NCT07654321"]
    assert metrics.extract_nct_ids("NCT123 is not a real id") == []


def test_citation_precision():
    assert metrics.citation_precision([], ["NCT01234567"]) is None
    assert metrics.citation_precision(["NCT01234567", "NCT09999999"], ["NCT01234567"]) == 0.5


def test_retrieval_recall_unlabelled_is_none():
    assert metrics.retrieval_recall(["NCT01234567"], []) is None
    assert metrics.retrieval_recall(["NCT01234567"], ["NCT01234567", "NCT02222222"]) == 0.5


def test_percentile():
    assert metrics.percentile([], 0.5) is None
    assert metrics.percentile([1, 2, 3, 4], 0.5) == 2.5


def test_meets_target_direction():
    assert metrics.meets_target("faithfulness", 0.95) is True
    assert metrics.meets_target("hallucination_rate", 0.10) is False
    assert metrics.meets_target("faithfulness", None) is None


# ----------------------------------------------------------------- judge

def _claim(label, cited=(), ok=None):
    return {"text": "x", "label": label, "cited_nct_ids": list(cited),
            "citation_supports_claim": ok, "reason": ""}


def test_scores_answerable():
    v = {"claims": [_claim("supported", ["NCT01234567"], True), _claim("unsupported"),
                    _claim("supported"), _claim("contradicted", ["NCT02222222"], False)],
         "abstained": False, "relevancy": 5}
    s = scores_from_verdict(v, should_abstain=False)
    assert s["faithfulness"] == 0.5
    assert s["hallucinated"] is True
    assert s["citation_support"] == 0.5
    assert s["answer_relevancy"] == 1.0
    assert s["abstained_correctly"] is None


def test_scores_clean_refusal():
    s = scores_from_verdict({"claims": [], "abstained": True, "relevancy": 1}, should_abstain=True)
    assert s["faithfulness"] is None and s["hallucinated"] is False
    assert s["abstained_correctly"] is True and s["answer_relevancy"] is None


def test_scores_refusal_that_still_invents_facts_is_wrong():
    v = {"claims": [_claim("unsupported")], "abstained": True, "relevancy": 2}
    assert scores_from_verdict(v, should_abstain=True)["abstained_correctly"] is False


class FakeAnthropic:
    def __init__(self, verdict):
        self.verdict, self.kwargs = verdict, None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input=self.verdict)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_creation_input_tokens=0, cache_read_input_tokens=800))


def test_claude_judge_forces_tool_and_caches_prompt():
    fake = FakeAnthropic({"claims": [], "abstained": True, "relevancy": 1})
    out = ClaudeJudge(model="claude-sonnet-5-5", client=fake).judge("q", "a", "ctx")
    assert fake.kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}
    assert fake.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert fake.kwargs["temperature"] == 0
    # 1000*$2 + 200*$10 + 800*$0.10 per MTok
    assert out["usage"]["cost_usd"] == pytest.approx((2000 + 2000 + 80) / 1e6)


# --------------------------------------------------------------- pricing

def test_unpriced_model_gives_none_not_zero():
    assert pricing.call_cost("grok-3-mini", 1000, 100) is None
    assert pricing.calls_cost([{"model": "claude-haiku-5-5", "input_tokens": 1_000_000,
                                "output_tokens": 0}]) == pytest.approx(0.10)


# ------------------------------------------------ Grok pipeline adapter

class FakeOpenAI:
    """Mimics the OpenAI client shape that src/agents.py uses."""

    def __init__(self, replies):
        self.replies = replies
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, max_tokens):
        prompt = messages[0]["content"]
        if "category word" in prompt:
            text = "trials"
        elif "CONTEXT:" in prompt:
            text = self.replies["synthesis"]
        else:
            text = "FDA adverse event data shows rash is most reported."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=len(prompt) // 4, completion_tokens=20))


@pytest.fixture
def grok_system(monkeypatch, tmp_path):
    import src.agents as agents
    from evals.systems import GrokPipelineSystem

    trials = [{"text": f"Title: trial {i}", "nct_id": f"NCT0000000{i}", "title": f"Trial {i}",
               "phase": "PHASE2", "score": 0.9} for i in range(1, 6)]
    monkeypatch.setattr(agents, "search", lambda *a, **k: trials)
    monkeypatch.setattr(agents, "query_faers",
                        lambda d: {"drug": d, "events": [{"reaction": "RASH", "count": 120}]})
    fake = FakeOpenAI({"synthesis": "NCT00000001 and NCT00000005 are Phase 2 trials."})
    monkeypatch.setattr(agents, "get_client", lambda: fake)
    return GrokPipelineSystem(collection=None, embed_model=None,
                              faers_snapshot=tmp_path / "faers.json")


def test_adapter_captures_exact_context_and_usage(grok_system, tmp_path):
    out = grok_system.answer("EGFR trials?", drug="osimertinib")
    # Synthesis only sees the top 3 trials, so the context must hold exactly those.
    assert out.context_nct_ids == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert out.retrieved_nct_ids == [f"NCT0000000{i}" for i in range(1, 6)]
    assert "RASH: 120 reports" in out.context_text
    assert [c["role"] for c in out.calls] == ["supervisor", "synthesis", "risk"]
    assert all(c["input_tokens"] > 0 and "prompt" not in c for c in out.calls)
    # FAERS response was frozen for the next run.
    assert json.loads((tmp_path / "faers.json").read_text())["osimertinib"]["events"]


def test_evaluate_one_flags_citation_outside_context(grok_system):
    q = {"id": "t-1", "category": "trials", "question": "EGFR trials?", "drug": "",
         "expect": {"should_abstain": False, "expected_nct_ids": ["NCT00000002"]}}
    rec = evaluate_one(q, grok_system, judge=None)
    # NCT00000005 was retrieved but never shown to the model: a fabricated citation.
    assert rec["metrics"]["fabricated_citations"] == ["NCT00000005"]
    assert rec["metrics"]["citation_precision"] == 0.5
    assert rec["metrics"]["retrieval_recall"] == 1.0
    assert rec["cost_usd"] is None  # grok is unpriced


def test_system_error_is_recorded_not_raised():
    class Boom:
        def answer(self, *a):
            raise TimeoutError("api down")
    rec = evaluate_one({"id": "x", "category": "trials", "question": "q", "expect": {}}, Boom(), None)
    assert rec["error"].startswith("system: TimeoutError")
    s = metrics.summarize([rec])
    assert s["n_errors"] == 1 and s["faithfulness"] is None


# ------------------------------------------------------- question set

def test_question_set_is_well_formed():
    qs = load_questions(ROOT / "evals" / "questions.jsonl")
    assert len(qs) >= 50
    cats = {q["category"] for q in qs}
    assert {"trials", "safety", "hybrid", "unanswerable", "false_premise"} <= cats
    for q in qs:
        assert q["question"].strip() and "should_abstain" in q["expect"]
        if q["category"] in ("safety", "hybrid"):
            assert q["drug"], q["id"]
        assert q["expect"]["should_abstain"] == (q["category"] == "unanswerable")


# ------------------------------------------------------------ dashboard

def test_dashboard_renders_sample_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "evals" / "dashboard.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Sample data" in w.value for w in at.warning)
    labels = [m.label for m in at.metric]
    assert "Faithfulness" in labels and "Hallucination rate" in labels

    at.multiselect[0].set_value(at.multiselect[0].options[:2]).run()
    assert not at.exception, at.exception
