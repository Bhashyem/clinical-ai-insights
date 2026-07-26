import os
from typing import TypedDict, List, Optional
from openai import OpenAI
from src.retriever import search, query_faers

def get_client():
    return OpenAI(
        api_key=os.environ["GROK_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

class ClinicalState(TypedDict):
    query:          str
    intent:         str
    trial_results:  List[dict]
    citations:      List[str]
    final_answer:   str
    error:          Optional[str]

def supervisor(state: ClinicalState) -> ClinicalState:
    client = get_client()
    prompt = f"""Classify this clinical research query into one of:
trials, evidence, safety, hybrid

Query: {state['query']}
Reply with only the category word."""
    resp = client.chat.completions.create(
        model="grok-3-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    intent = resp.choices[0].message.content.strip().lower()
    if intent not in ["trials", "evidence", "safety", "hybrid"]:
        intent = "hybrid"
    state["intent"] = intent
    return state

def trial_agent(state: ClinicalState, collection, embed_model) -> ClinicalState:
    results = search(state["query"], collection, embed_model, n_results=5)
    state["trial_results"] = results
    state["citations"] = [r["nct_id"] for r in results if r["nct_id"]]
    return state

def synthesis_agent(state: ClinicalState) -> ClinicalState:
    client = get_client()
    context_parts = []
    for trial in state["trial_results"][:3]:
        context_parts.append(
            f"Trial {trial['nct_id']}: {trial['title']}\n"
            f"Phase: {trial['phase']}\n"
            f"Excerpt: {trial['text'][:200]}"
        )
    context = "\n\n".join(context_parts)
    prompt = f"""You are a clinical research assistant.
Answer using ONLY the trials below. Cite NCT IDs. Be concise (3-5 sentences).

CONTEXT:
{context}

QUESTION: {state['query']}"""
    resp = client.chat.completions.create(
        model="grok-3-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    state["final_answer"] = resp.choices[0].message.content.strip()
    return state

def risk_agent(state: ClinicalState, drug_name: str) -> ClinicalState:
    client = get_client()
    faers = query_faers(drug_name)
    if not faers["events"]:
        return state
    ae_lines = "\n".join([f"  - {e['reaction']}: {e['count']:,} reports"
                          for e in faers["events"][:5]])
    prompt = f"""Append a 1-sentence safety note starting with 'FDA adverse event data shows...'

Existing answer: {state['final_answer']}
FDA data for {drug_name}:
{ae_lines}"""
    resp = client.chat.completions.create(
        model="grok-3-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )
    state["final_answer"] += "\n\n" + resp.choices[0].message.content.strip()
    return state

def run_pipeline(query: str, collection, embed_model,
                 drug_name: str = "") -> dict:
    state: ClinicalState = {
        "query": query, "intent": "", "trial_results": [],
        "citations": [], "final_answer": "", "error": None,
    }
    state = supervisor(state)
    state = trial_agent(state, collection, embed_model)
    state = synthesis_agent(state)
    if drug_name:
        state = risk_agent(state, drug_name)
    return {
        "question":  state["query"],
        "intent":    state["intent"],
        "answer":    state["final_answer"],
        "citations": state["citations"],
        "trials":    state["trial_results"],
    }
