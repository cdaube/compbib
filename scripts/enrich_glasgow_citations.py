"""
Enrich Glasgow researcher abstracts with citation counts from OpenAlex.

Mirrors scripts/enrich_citations.py but targets data/glasgow_abstracts.csv.

Usage:
    uv run python scripts/enrich_glasgow_citations.py
"""

import pandas as pd
import requests
import time
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_CSV = os.path.join(DATA_DIR, "glasgow_abstracts.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "glasgow_citations.csv")

OPENALEX_BASE = "https://api.openalex.org/works"
BATCH_SIZE = 50
POLITE_EMAIL = "christoph.daube@gmail.com"


def fetch_citations_batch(pmids: list[str]) -> dict[str, int]:
    """Fetch citation counts for a batch of PMIDs from OpenAlex."""
    filter_str = "|".join(pmids)
    params = {
        "filter": f"ids.pmid:{filter_str}",
        "select": "ids,cited_by_count",
        "per-page": BATCH_SIZE,
        "mailto": POLITE_EMAIL,
    }

    resp = requests.get(OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = {}
    for work in data.get("results", []):
        pmid_url = work.get("ids", {}).get("pmid", "")
        pmid = pmid_url.rstrip("/").split("/")[-1] if pmid_url else None
        if pmid:
            results[pmid] = work.get("cited_by_count", 0)

    return results


def fetch_citations_by_openalex_ids(openalex_ids: list[str]) -> dict[str, int]:
    """Fetch citation counts keyed by OpenAlex work ID."""
    results = {}
    for openalex_id in openalex_ids:
        oid = str(openalex_id).rstrip("/").split("/")[-1]
        if not oid:
            continue
        params = {"select": "id,cited_by_count", "mailto": POLITE_EMAIL}
        resp = requests.get(f"{OPENALEX_BASE}/{oid}", params=params, timeout=30)
        resp.raise_for_status()
        work = resp.json()
        results[oid] = work.get("cited_by_count", 0)
        time.sleep(0.05)
    return results


def enrich():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} abstracts from {INPUT_CSV}")

    id_col = "paper_id" if "paper_id" in df.columns else "pmid"
    has_openalex = "openalex_id" in df.columns
    df[id_col] = df[id_col].astype(str)
    if has_openalex:
        df["openalex_id"] = df["openalex_id"].astype(str)
    ids = df[id_col].tolist()

    # Resume support
    already_done = set()
    existing_rows = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        existing_id_col = "paper_id" if "paper_id" in existing.columns else "pmid"
        already_done = set(existing[existing_id_col].astype(str))
        existing_rows = existing.to_dict("records")
        print(f"Resuming: {len(already_done)} papers already fetched.")

    remaining = [p for p in ids if p not in already_done]
    print(f"Fetching citations for {len(remaining)} papers...")

    all_results = list(existing_rows)
    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    for batch in tqdm(batches, desc="OpenAlex"):
        for attempt in range(3):
            try:
                if has_openalex:
                    openalex_lookup = (
                        df[df[id_col].isin(batch)]
                        .set_index(id_col)["openalex_id"]
                        .astype(str)
                        .to_dict()
                    )
                    oa_ids = [openalex_lookup[p] for p in batch if openalex_lookup.get(p)]
                    citations = fetch_citations_by_openalex_ids(oa_ids)
                    for paper_id in batch:
                        oid = openalex_lookup.get(paper_id, "")
                        oid = str(oid).rstrip("/").split("/")[-1]
                        all_results.append({
                            id_col: paper_id,
                            "openalex_id": oid,
                            "cited_by_count": citations.get(oid, 0),
                        })
                else:
                    citations = fetch_citations_batch(batch)
                    for pmid in batch:
                        all_results.append({
                            "pmid": pmid,
                            "cited_by_count": citations.get(pmid, 0),
                        })
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if len(all_results) % (BATCH_SIZE * 100) < BATCH_SIZE:
            pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

        time.sleep(0.1)

    out = pd.DataFrame(all_results)
    out.drop_duplicates(subset=id_col if id_col in out.columns else "pmid", inplace=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Citation data for {len(out)} papers saved to {OUTPUT_CSV}")
    print(f"Mean citations: {out['cited_by_count'].mean():.1f}")
    print(f"Median citations: {out['cited_by_count'].median():.0f}")
    print(f"Max citations: {out['cited_by_count'].max()}")


if __name__ == "__main__":
    enrich()
