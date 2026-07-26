import os
import sys
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ── Add src to path so imports work
sys.path.insert(0, os.path.dirname(__file__))
from src.etl import fetch_clinical_trials, ingest_trials
from src.agents import run_pipeline

# ── Page config
st.set_page_config(
    page_title="ClinicalAI Insights",
    page_icon="🔬",
    layout="centered"
)

# ── Load API key (Streamlit Cloud uses st.secrets, Docker uses env var)
def get_api_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return os.environ.get("GROK_API_KEY", "")

os.environ["GROK_API_KEY"] = get_api_key()

# ── Load model + ingest data ONCE (cached across all users)
@st.cache_resource
def load_system():
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    chroma = chromadb.Client()

    try:
        collection = chroma.get_collection("clinical_trials")
    except:
        collection = chroma.create_collection("clinical_trials")
        conditions = ["lung cancer", "non-small cell lung cancer",
                      "EGFR mutation", "KRAS mutation cancer"]
        for condition in conditions:
            trials = fetch_clinical_trials(condition=condition, max_studies=40)
            ingest_trials(trials, collection, embed_model)

    return embed_model, collection

# ── UI
st.title("🔬 ClinicalAI Insights Platform")
st.caption("Multi-agent clinical trial intelligence · Live data from ClinicalTrials.gov + FDA FAERS")
st.divider()

# Load system with spinner
with st.spinner("Loading models and ingesting live trial data (first run ~60s)..."):
    embed_model, collection = load_system()

st.success(f"System ready · {collection.count()} trial chunks indexed")
st.divider()

# ── Query form
query = st.text_input(
    "Ask a clinical research question",
    placeholder="e.g. What Phase 2 trials are recruiting for EGFR lung cancer?",
)
drug_name = st.text_input(
    "Drug name for FDA safety data (optional)",
    placeholder="e.g. osimertinib",
)

run_btn = st.button("Search", type="primary", use_container_width=True)

# ── Run pipeline
if run_btn and query:
    with st.spinner("Running agents..."):
        result = run_pipeline(
            query=query,
            collection=collection,
            embed_model=embed_model,
            drug_name=drug_name.strip(),
        )

    st.divider()

    # Answer
    st.subheader("Answer")
    st.write(result["answer"])

    # Citations
    if result["citations"]:
        st.subheader("Citations")
        for nct_id in result["citations"]:
            url = f"https://clinicaltrials.gov/study/{nct_id}"
            st.markdown(f"• [{nct_id}]({url})")

    # Trial details expander
    if result["trials"]:
        with st.expander("View retrieved trials"):
            for trial in result["trials"]:
                st.markdown(f"**{trial['nct_id']}** — {trial['title']}")
                st.caption(f"Phase: {trial['phase']} · Relevance: {trial['score']}")
                st.divider()

elif run_btn and not query:
    st.warning("Please enter a question.")
