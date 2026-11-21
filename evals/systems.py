"""Adapters that put each system under test behind one interface.

Every system returns a SystemOutput. The harness never needs to know whether
the answer came from the Grok/LangGraph pipeline or the Claude rebuild, so the
two can be compared question by question on the same snapshot.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from evals.metrics import NCT_RE


@dataclass
class SystemOutput:
    answer: str
    # Exactly the evidence the answering model saw. Faithfulness and citation
    # checks are judged against this, not against everything retrieved.
    context_text: str
    context_nct_ids: list[str]
    # Everything the retriever returned (for retrieval recall).
    retrieved_nct_ids: list[str]
    # One entry per model call: model, input_tokens, output_tokens,
    # cache_write_tokens, cache_read_tokens, latency_s, role.
    calls: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Grok / LangGraph pipeline (src/agents.py), instrumented without changing it
# --------------------------------------------------------------------------

_local = threading.local()


class _RecordingCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        t0 = time.perf_counter()
        resp = self._inner.create(**kwargs)
        dt = time.perf_counter() - t0
        rec = getattr(_local, "calls", None)
        if rec is not None:
            usage = getattr(resp, "usage", None)
            prompt = "\n".join(str(m.get("content", "")) for m in kwargs.get("messages", []))
            rec.append({
                "model": kwargs.get("model", ""),
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "cache_write_tokens": 0,
                "cache_read_tokens": 0,
                "latency_s": dt,
                "prompt": prompt,
            })
        return resp


class _RecordingChat:
    def __init__(self, inner):
        self.completions = _RecordingCompletions(inner.completions)


class _RecordingClient:
    def __init__(self, inner):
        self._inner = inner
        self.chat = _RecordingChat(inner.chat)


def _role_of(prompt: str) -> str:
    if "Reply with only the category word" in prompt:
        return "supervisor"
    if "CONTEXT:" in prompt:
        return "synthesis"
    if "FDA data for" in prompt:
        return "risk"
    return "other"


def _between(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return (text[i:j] if j >= 0 else text[i:]).strip()


class CachedFaers:
    """Freezes FAERS responses so repeated runs see identical safety data."""

    def __init__(self, path: Path, live_fn):
        self.path = path
        self.live_fn = live_fn
        self.lock = threading.Lock()
        self.cache = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, drug_name: str):
        key = drug_name.strip().lower()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        result = self.live_fn(drug_name)
        if not result.get("error"):
            with self.lock:
                self.cache[key] = result
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.cache, indent=2))
        return result


class GrokPipelineSystem:
    """Runs src.agents.run_pipeline and records every model call it makes."""

    name = "grok-langgraph"

    def __init__(self, collection, embed_model, faers_snapshot: Path | None = None):
        import src.agents as agents

        self.agents = agents
        self.collection = collection
        self.embed_model = embed_model
        original_get_client = agents.get_client
        agents.get_client = lambda: _RecordingClient(original_get_client())
        if faers_snapshot is not None:
            agents.query_faers = CachedFaers(faers_snapshot, agents.query_faers)

    def answer(self, question: str, drug: str = "") -> SystemOutput:
        _local.calls = []
        t0 = time.perf_counter()
        try:
            result = self.agents.run_pipeline(
                query=question, collection=self.collection,
                embed_model=self.embed_model, drug_name=drug,
            )
        finally:
            calls, _local.calls = _local.calls, None
        latency = time.perf_counter() - t0

        context_parts = []
        for c in calls:
            c["role"] = _role_of(c["prompt"])
            if c["role"] == "synthesis":
                context_parts.append(_between(c["prompt"], "CONTEXT:", "QUESTION:"))
            elif c["role"] == "risk":
                context_parts.append("FDA data for " + _between(c["prompt"], "FDA data for", None))
            c.pop("prompt")
        context_text = "\n\n".join(p for p in context_parts if p)

        return SystemOutput(
            answer=result["answer"],
            context_text=context_text,
            context_nct_ids=sorted(set(NCT_RE.findall(context_text))),
            retrieved_nct_ids=[t["nct_id"] for t in result.get("trials", []) if t.get("nct_id")],
            calls=calls,
            latency_s=latency,
            extra={"intent": result.get("intent", "")},
        )


# --------------------------------------------------------------------------
# Claude rebuild (Project #1) - plug it in here when it exists
# --------------------------------------------------------------------------

class ClaudeAgentSystem:
    """Placeholder for the Claude Agent SDK + MCP rebuild.

    To plug it in, implement answer() so that it returns a SystemOutput:
      - context_text: the tool results the model actually read (trial records
        and FAERS counts returned by the MCP servers), concatenated.
      - context_nct_ids: NCT IDs present in those tool results.
      - retrieved_nct_ids: every NCT ID any search tool returned.
      - calls: one dict per Messages API call, using response.usage:
        input_tokens, output_tokens, cache_creation_input_tokens
        (-> cache_write_tokens), cache_read_input_tokens (-> cache_read_tokens).
    Everything else (judging, metrics, dashboard) works unchanged.
    """

    name = "claude-agent-sdk"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The Claude rebuild is Project #1. Implement ClaudeAgentSystem.answer() "
            "in evals/systems.py once it exists."
        )

    def answer(self, question: str, drug: str = "") -> SystemOutput:  # pragma: no cover
        raise NotImplementedError
