import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "]
)

def fetch_clinical_trials(condition="lung cancer", max_studies=50):
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
        "pageSize": min(max_studies, 100),
        "fields": "NCTId,BriefTitle,BriefSummary,Phase,OverallStatus,Condition,InterventionName,LeadSponsorName",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    studies = resp.json().get("studies", [])
    records = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_mod     = proto.get("identificationModule", {})
        desc_mod   = proto.get("descriptionModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        sponsor_mod= proto.get("sponsorCollaboratorsModule", {})
        arms_mod   = proto.get("armsInterventionsModule", {})
        interventions = [i.get("interventionName", "") for i in arms_mod.get("interventions", [])]
        phases = design_mod.get("phases", [])
        records.append({
            "nct_id":        id_mod.get("nctId", ""),
            "title":         id_mod.get("briefTitle", ""),
            "summary":       desc_mod.get("briefSummary", ""),
            "phase":         ", ".join(phases) if phases else "N/A",
            "status":        status_mod.get("overallStatus", ""),
            "condition":     condition,
            "interventions": ", ".join(interventions[:3]),
            "sponsor":       sponsor_mod.get("leadSponsor", {}).get("leadSponsorName", ""),
        })
    return records


def ingest_trials(trials, collection, embed_model):
    all_chunks, all_ids, all_metadata = [], [], []
    for trial in trials:
        text = f"""Title: {trial['title']}
Phase: {trial['phase']}
Status: {trial['status']}
Condition: {trial['condition']}
Interventions: {trial['interventions']}
Summary: {trial['summary']}"""
        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{trial['nct_id']}_chunk_{j}")
            all_metadata.append({
                "nct_id":    trial['nct_id'],
                "title":     trial['title'][:80],
                "phase":     trial['phase'],
                "condition": trial['condition'],
                "source":    "clinicaltrials.gov",
            })
    embeddings = embed_model.encode(all_chunks, show_progress_bar=False).tolist()
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )
    return len(all_chunks)
