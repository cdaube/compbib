"""
Build a citation graph among Glasgow researcher papers using OpenAlex.

For each paper in glasgow_abstracts.csv, fetches its references from OpenAlex,
then filters to only keep edges where both citing and cited paper are in our dataset.

Outputs:
  data/glasgow_citation_graph.csv  — edge list between dataset paper IDs

Usage:
    uv run python scripts/enrich_glasgow_citation_graph.py
"""

import pandas as pd
import requests
import time
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_CSV = os.path.join(DATA_DIR, "glasgow_abstracts.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "glasgow_citation_graph.csv")
OPENALEX_MAP_CACHE = os.path.join(DATA_DIR, ".glasgow_openalex_map.csv")

OPENALEX_BASE = "https://api.openalex.org/works"
BATCH_SIZE = 50
POLITE_EMAIL = "christoph.daube@gmail.com"


def fetch_batch(pmids: list[str]) -> list[dict]:
    """Fetch OpenAlex IDs and referenced_works for a batch of PMIDs."""
    filter_str = "|".join(pmids)
    params = {
        "filter": f"ids.pmid:{filter_str}",
        "select": "id,ids,referenced_works",
        "per-page": BATCH_SIZE,
        "mailto": POLITE_EMAIL,
    }
    resp = requests.get(OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for work in data.get("results", []):
        pmid_url = work.get("ids", {}).get("pmid", "")
        pmid = pmid_url.rstrip("/").split("/")[-1] if pmid_url else None
        openalex_id = work.get("id", "")
        refs = work.get("referenced_works", []) or []
        if pmid:
            results.append({
                "pmid": pmid,
                "openalex_id": openalex_id,
                "referenced_works": refs,
            })
    return results


def fetch_openalex_work(paper_id: str, openalex_id: str) -> dict:
    """Fetch referenced_works for a known OpenAlex work ID."""
    oid = str(openalex_id or "").rstrip("/").split("/")[-1]
    params = {"select": "id,referenced_works", "mailto": POLITE_EMAIL}
    resp = requests.get(f"{OPENALEX_BASE}/{oid}", params=params, timeout=30)
    resp.raise_for_status()
    work = resp.json()
    return {
        "paper_id": paper_id,
        "openalex_id": work.get("id", ""),
        "referenced_works": work.get("referenced_works", []) or [],
    }


def build_graph():
    df = pd.read_csv(INPUT_CSV)
    id_col = "paper_id" if "paper_id" in df.columns else "pmid"
    df[id_col] = df[id_col].astype(str)
    paper_ids = df[id_col].tolist()
    has_openalex = "openalex_id" in df.columns
    if has_openalex:
        df["openalex_id"] = df["openalex_id"].astype(str)
    print(f"Loaded {len(paper_ids)} papers from {INPUT_CSV}")

    if has_openalex and "referenced_works" in df.columns:
        print("Using referenced_works from abstracts CSV.")
        oa_to_paper_id = {
            str(row["openalex_id"]).rstrip("/").split("/")[-1]: row[id_col]
            for _, row in df.iterrows()
            if str(row.get("openalex_id", "")).strip()
        }
        our_oa_urls = {f"https://openalex.org/{oid}": paper_id for oid, paper_id in oa_to_paper_id.items()}
        edges = []
        for _, row in df.iterrows():
            paper_id = row[id_col]
            refs = str(row.get("referenced_works") or "")
            if refs.lower() == "nan":
                refs = ""
            for ref_oa_id in refs.split("|"):
                cited_paper_id = our_oa_urls.get(ref_oa_id)
                if cited_paper_id and cited_paper_id != paper_id:
                    edges.append({
                        "citing_paper_id": paper_id,
                        "cited_paper_id": cited_paper_id,
                    })
        edge_df = pd.DataFrame(edges).drop_duplicates()
        edge_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nDone! {len(edge_df)} citation edges (within dataset) saved to {OUTPUT_CSV}")
        return

    # Step 1: Fetch OpenAlex IDs + references for all papers
    if os.path.exists(OPENALEX_MAP_CACHE):
        print("Loading cached OpenAlex mappings...")
        cached = pd.read_csv(OPENALEX_MAP_CACHE)
        cache_id_col = "paper_id" if "paper_id" in cached.columns else "pmid"
        already_done = set(cached[cache_id_col].astype(str))
        all_records = cached.to_dict("records")
        # Parse stringified lists back
        for r in all_records:
            if isinstance(r["referenced_works"], str):
                r["referenced_works"] = r["referenced_works"].split("|") if r["referenced_works"] else []
    else:
        already_done = set()
        all_records = []

    remaining = [p for p in paper_ids if p not in already_done]
    print(f"Fetching OpenAlex data for {len(remaining)} papers...")

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    for batch in tqdm(batches, desc="OpenAlex refs"):
        for attempt in range(3):
            try:
                if has_openalex:
                    openalex_lookup = (
                        df[df[id_col].isin(batch)]
                        .set_index(id_col)["openalex_id"]
                        .astype(str)
                        .to_dict()
                    )
                    results = [
                        fetch_openalex_work(paper_id, openalex_lookup[paper_id])
                        for paper_id in batch
                        if openalex_lookup.get(paper_id)
                    ]
                else:
                    results = fetch_batch(batch)
                all_records.extend(results)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        # Checkpoint every 200 batches
        if len(all_records) % (BATCH_SIZE * 200) < BATCH_SIZE:
            _save_cache(all_records)

        time.sleep(0.1)

    _save_cache(all_records)

    # Step 2: Build mapping from OpenAlex ID to dataset paper ID.
    oa_to_paper_id = {}
    paper_refs = {}
    for rec in all_records:
        oa_id = rec["openalex_id"]
        paper_id = rec.get("paper_id") or rec.get("pmid")
        oa_to_paper_id[oa_id] = paper_id
        paper_refs[paper_id] = rec["referenced_works"]

    print(f"Mapped {len(oa_to_paper_id)} papers to OpenAlex IDs")

    # Step 3: Build edge list — only keep edges where both ends are in our dataset
    our_oa_ids = set(oa_to_paper_id.keys())
    edges = []
    for paper_id, refs in paper_refs.items():
        for ref_oa_id in refs:
            if ref_oa_id in our_oa_ids:
                cited_paper_id = oa_to_paper_id[ref_oa_id]
                if cited_paper_id != paper_id:  # no self-citations
                    edges.append({
                        "citing_paper_id": paper_id,
                        "cited_paper_id": cited_paper_id,
                    })

    edge_df = pd.DataFrame(edges).drop_duplicates()
    edge_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! {len(edge_df)} citation edges (within dataset) saved to {OUTPUT_CSV}")

    # Clean up cache
    if os.path.exists(OPENALEX_MAP_CACHE):
        os.remove(OPENALEX_MAP_CACHE)


def _save_cache(records):
    """Save intermediate results with referenced_works as pipe-separated strings."""
    rows = []
    for r in records:
        rows.append({
            "paper_id": r.get("paper_id") or r.get("pmid"),
            "openalex_id": r["openalex_id"],
            "referenced_works": "|".join(r["referenced_works"]) if r["referenced_works"] else "",
        })
    pd.DataFrame(rows).to_csv(OPENALEX_MAP_CACHE, index=False)


if __name__ == "__main__":
    build_graph()
