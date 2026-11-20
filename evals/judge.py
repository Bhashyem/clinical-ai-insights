"""LLM-as-judge using Claude with a forced tool call for structured verdicts.

One judge call per question. The judge splits the answer into atomic claims
and labels each one against the exact context the answering model saw. All
scores are then computed in Python from those labels, so the arithmetic is
deterministic and auditable on the dashboard.

Use a judge from a different model family than the system under test where
you can. When you evaluate the Claude rebuild, run a second pass with another
judge model and report both, so the numbers are not Claude grading Claude.
"""
from __future__ import annotations

import time

from evals.pricing import call_cost

JUDGE_SYSTEM = """You are a strict evaluator of a clinical-research question answering system.
You judge ONLY against the CONTEXT provided, never against your own medical knowledge:
a claim that is true in the real world but absent from the context is UNSUPPORTED.

Procedure:
1. Split the ANSWER into atomic factual claims (one fact each). Ignore pure
   hedges, greetings and statements that the system lacks information.
2. Label each claim:
   - supported: the context states it or directly implies it
   - unsupported: the context does not contain it
   - contradicted: the context says otherwise
3. For each claim, list the NCT IDs the answer attaches to it. If it attaches
   any, say whether those specific trials' context supports the claim.
4. abstained: true if the answer mainly declines or says the information is
   not available, rather than answering.
5. relevancy (1-5): how directly the answer addresses the QUESTION, regardless
   of correctness. 1 = off-topic, 5 = fully on point.
Always call the record_verdict tool."""

VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record the evaluation of one answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string",
                                  "enum": ["supported", "unsupported", "contradicted"]},
                        "cited_nct_ids": {"type": "array", "items": {"type": "string"}},
                        "citation_supports_claim": {"type": ["boolean", "null"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["text", "label", "cited_nct_ids",
                                 "citation_supports_claim", "reason"],
                },
            },
            "abstained": {"type": "boolean"},
            "relevancy": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["claims", "abstained", "relevancy"],
    },
}


class ClaudeJudge:
    def __init__(self, model: str = "claude-sonnet-5-5", client=None, max_retries: int = 3):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model
        self.max_retries = max_retries

    def judge(self, question: str, answer: str, context_text: str) -> dict:
        user = (f"QUESTION:\n{question}\n\n"
                f"CONTEXT (everything the system saw):\n{context_text or '(empty)'}\n\n"
                f"ANSWER:\n{answer}")
        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0,
                    # The system prompt and tool are identical on every call,
                    # so cache them: cheaper and faster from the second call on.
                    system=[{"type": "text", "text": JUDGE_SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    tools=[VERDICT_TOOL],
                    tool_choice={"type": "tool", "name": "record_verdict"},
                    messages=[{"role": "user", "content": user}],
                )
                dt = time.perf_counter() - t0
                block = next(b for b in resp.content if getattr(b, "type", "") == "tool_use")
                u = resp.usage
                usage = {
                    "model": self.model,
                    "input_tokens": getattr(u, "input_tokens", 0) or 0,
                    "output_tokens": getattr(u, "output_tokens", 0) or 0,
                    "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                    "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                    "latency_s": dt,
                }
                usage["cost_usd"] = call_cost(
                    self.model, usage["input_tokens"], usage["output_tokens"],
                    usage["cache_write_tokens"], usage["cache_read_tokens"])
                return {"verdict": dict(block.input), "usage": usage}
            except Exception as e:  # network blips, overloaded, malformed output
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"judge failed after {self.max_retries} attempts: {last_err}")


def scores_from_verdict(verdict: dict, should_abstain: bool) -> dict:
    """Turn the judge's labels into numbers. Pure function, unit-tested."""
    claims = verdict.get("claims", [])
    n = len(claims)
    supported = sum(c["label"] == "supported" for c in claims)
    bad = n - supported
    cited = [c for c in claims if c.get("cited_nct_ids")]
    cited_ok = [c for c in cited if c.get("citation_supports_claim") is True]
    abstained = bool(verdict.get("abstained"))

    out = {
        "n_claims": n,
        "n_unsupported": bad,
        "abstained": abstained,
        # Faithfulness is undefined for an answer with no claims (e.g. a clean refusal).
        "faithfulness": (supported / n) if n else None,
        "hallucinated": bad > 0,
        "citation_support": (len(cited_ok) / len(cited)) if cited else None,
        "answer_relevancy": (verdict["relevancy"] - 1) / 4 if "relevancy" in verdict else None,
    }
    if should_abstain:
        # Correct only if it declined AND did not slip in made-up facts.
        out["abstained_correctly"] = abstained and bad == 0
        # Relevancy is not meaningful when the right answer is "I don't know".
        out["answer_relevancy"] = None
    else:
        out["abstained_correctly"] = None
    return out
