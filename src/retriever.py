def search(query, collection, embed_model, n_results=5):
    query_vector = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    seen = set()
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        nct_id = meta.get("nct_id", "")
        if nct_id not in seen:
            seen.add(nct_id)
            retrieved.append({
                "text":   doc,
                "nct_id": nct_id,
                "title":  meta.get("title", ""),
                "phase":  meta.get("phase", ""),
                "score":  round(1 - dist, 3),
            })
    return retrieved


def query_faers(drug_name):
    import requests
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "count":  "patient.reaction.reactionmeddrapt.exact",
        "limit":  8,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return {"drug": drug_name,
                "events": [{"reaction": r["term"], "count": r["count"]} for r in results]}
    except Exception as e:
        return {"drug": drug_name, "events": [], "error": str(e)}
