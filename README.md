# ClinicalAI Insights Platform

A production-grade multi-agent AI system for clinical trial intelligence 
built on live data from ClinicalTrials.gov and FDA FAERS.

## What it does
A biopharma research scientist asks: *"What Phase 2 trials are recruiting 
for EGFR lung cancer and what are the safety signals?"*  
The system answers in under 10 seconds with real NCT citations and FDA 
adverse event data — no hallucination.

## Evaluation Results
| Metric | Score | Target |
|---|---|---|
| Faithfulness | 0.883 | >0.90 |
| Answer Relevancy | 0.900 | >0.85 |

## Agent Architecture
- **Supervisor** — classifies query intent and routes to correct agent
- **Trial Agent** — semantic search over 200+ live clinical trials
- **Synthesis Agent** — grounded answer generation with NCT citations
- **Risk Agent** — FDA FAERS adverse event data per drug

## Tech Stack
Grok API (xAI) · ChromaDB · sentence-transformers · LangGraph ·  
ClinicalTrials.gov REST API v2 · FDA OpenFDA API · Python 3.12

## Why RAG over Fine-tuning
Clinical trial data updates daily — fine-tuning bakes stale knowledge 
into weights and costs $500–5,000 per retrain. RAG with live API 
ingestion stays current at ~$0.01/query with full source attribution.

## Domain
Healthcare AI / Biopharma — oncology focus (lung cancer, EGFR, KRAS, 
amivantamab). Built on 7+ years of biopharma AI delivery experience 
across Roche/Genentech and healthcare clients.
