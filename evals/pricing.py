"""Per-million-token prices in USD, used to compute cost per query.

Claude prices are from https://platform.claude.com/docs/en/about-claude/pricing
(checked Nov 2026). Prices change: re-check before publishing numbers.

Grok prices are left as None on purpose: fill them in from x.ai's pricing page.
While a model's price is None, its cost shows as "n/a" instead of a wrong number.
"""
from __future__ import annotations

# model: input, output, 5-minute cache write, cache read
PRICES: dict[str, dict[str, float | None]] = {
    "claude-opus-5-5":   {"input": 4.00, "output": 20.00, "cache_write": 5.00,  "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50,  "cache_read": 0.10},
    # Haiku 5.5 at prompts <=100K tokens; this project's prompts are far smaller.
    "claude-haiku-5-5":  {"input": 0.10, "output": 0.50,  "cache_write": 0.125, "cache_read": 0.01},
    "grok-3-mini":       {"input": None, "output": None,  "cache_write": None,  "cache_read": None},
}


def call_cost(model: str, input_tokens: int, output_tokens: int,
              cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> float | None:
    p = PRICES.get(model)
    if not p or p["input"] is None or p["output"] is None:
        return None
    cost = input_tokens * p["input"] + output_tokens * p["output"]
    if cache_write_tokens:
        if p["cache_write"] is None:
            return None
        cost += cache_write_tokens * p["cache_write"]
    if cache_read_tokens:
        if p["cache_read"] is None:
            return None
        cost += cache_read_tokens * p["cache_read"]
    return cost / 1_000_000


def calls_cost(calls: list[dict]) -> float | None:
    """Total cost of a list of recorded calls, or None if any call is unpriced."""
    total = 0.0
    for c in calls:
        cc = call_cost(c["model"], c.get("input_tokens", 0), c.get("output_tokens", 0),
                       c.get("cache_write_tokens", 0), c.get("cache_read_tokens", 0))
        if cc is None:
            return None
        total += cc
    return total
