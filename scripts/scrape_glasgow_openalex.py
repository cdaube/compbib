"""
OpenAlex scraper for Glasgow University researchers in the imaging spreadsheet.

This keeps the existing spreadsheet roster from scrape_glasgow.py, resolves each
person to an OpenAlex author, then fetches works by author ID. That captures
papers outside PubMed, including computing science and engineering outputs.

Outputs:
  data/glasgow_abstracts.csv          one row per unique OpenAlex work
  data/glasgow_authors.csv            spreadsheet author to paper mapping
  data/glasgow_citations.csv          citation counts from OpenAlex
  data/glasgow_author_candidates.csv  author-resolution audit trail

Usage:
    uv run python scripts/scrape_glasgow_openalex.py --overwrite

Optional:
  data/glasgow_author_overrides.csv with researcher_name,openalex_author_id
  can be used to resolve ambiguous people from glasgow_author_candidates.csv.
  Optional columns profile_publications_url and doi_allowlist restrict works
  for fragile/common-name author records.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

try:
    from scrape_glasgow import XLSX_FILE, parse_researchers
except ModuleNotFoundError:
    from scripts.scrape_glasgow import XLSX_FILE, parse_researchers


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "glasgow_abstracts.csv")
AUTHORS_CSV = os.path.join(DATA_DIR, "glasgow_authors.csv")
CITATIONS_CSV = os.path.join(DATA_DIR, "glasgow_citations.csv")
CANDIDATES_CSV = os.path.join(DATA_DIR, "glasgow_author_candidates.csv")
OVERRIDES_CSV = os.path.join(DATA_DIR, "glasgow_author_overrides.csv")

OPENALEX_BASE = "https://api.openalex.org"
POLITE_EMAIL = "christoph.daube@gmail.com"
GLASGOW_ROR = "https://ror.org/00vtgdb53"

REQUEST_SLEEP = 0.1
MAX_RETRIES = 5
WORKS_PER_PAGE = 200
AUTHOR_CANDIDATES = 10


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace(".", " ")
    value = re.sub(r"[^a-zA-Z\s-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def clean_researcher_name(value: str) -> str:
    value = str(value or "").strip().replace(".", " ")
    value = re.sub(r"\s+", " ", value)
    aliases = {
        "guillaume rousselet": "Guillaume Rousselet",
        "rosario lopezgonzalez": "Rosario Lopez Gonzalez",
        "pauline hallbarrientos": "Pauline Hall Barrientos",
        "cristina gonzalezgarcia": "Cristina Gonzalez Garcia",
        "joshuafranz einsle": "Joshua Franz Einsle",
        "sarah allwood": "Sarah Allwood-Spiers",
        "shajan gunamony": "Gunamony Shajan",
        "mick craig": "Michael T. Craig",
    }
    return aliases.get(normalize_name(value), value)


def clean_researchers(researchers: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = []
    seen = set()
    for researcher in researchers:
        row = dict(researcher)
        row["name"] = clean_researcher_name(row["name"])
        key = (normalize_name(row["name"]), row["school"], row["college"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)
    return cleaned


def short_openalex_id(value: str) -> str:
    return str(value or "").rstrip("/").split("/")[-1]


def clean_doi(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"^https?://doi\.org/", "", str(value), flags=re.I).strip()


def normalize_doi(value: str | None) -> str:
    doi = clean_doi(value)
    if doi.lower() == "nan":
        return ""
    doi = doi.strip().strip(".,;)'\"")
    return doi.lower()


def pmid_from_ids(ids: dict[str, Any]) -> str:
    pmid = ids.get("pmid", "")
    return str(pmid).rstrip("/").split("/")[-1] if pmid else ""


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, offsets in index.items():
        for offset in offsets:
            positions.append((int(offset), word))
    positions.sort(key=lambda item: item[0])
    return html.unescape(" ".join(word for _, word in positions))


def fetch_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = path if path.startswith("http") else f"{OPENALEX_BASE}{path}"
    params = dict(params or {})
    params.setdefault("mailto", POLITE_EMAIL)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=45)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def glasgow_institution_id() -> str:
    data = fetch_json(f"/institutions/ror:{GLASGOW_ROR}")
    return short_openalex_id(data["id"])


def institution_names(candidate: dict[str, Any]) -> list[str]:
    names = []
    for inst in candidate.get("last_known_institutions") or []:
        names.append(str(inst.get("display_name", "")))
        names.append(str(inst.get("ror", "")))
        names.append(short_openalex_id(inst.get("id", "")))
    for aff in candidate.get("affiliations") or []:
        institutions = aff.get("institutions")
        if institutions is None and aff.get("institution"):
            institutions = [aff.get("institution")]
        for inst in institutions or []:
            names.append(str(inst.get("display_name", "")))
            names.append(str(inst.get("ror", "")))
            names.append(short_openalex_id(inst.get("id", "")))
    return [n for n in names if n]


def has_glasgow_affiliation(candidate: dict[str, Any], glasgow_id: str) -> bool:
    names = institution_names(candidate)
    joined = " | ".join(names).lower()
    return (
        glasgow_id in names
        or GLASGOW_ROR.lower() in joined
        or "university of glasgow" in joined
    )


def name_score(query_name: str, candidate_name: str) -> float:
    query = normalize_name(query_name)
    candidate = normalize_name(candidate_name)
    ratio = SequenceMatcher(None, query, candidate).ratio()
    q_parts = query.split()
    c_parts = candidate.split()
    score = ratio * 70
    if query == candidate:
        score += 25
    if q_parts and c_parts and q_parts[-1] == c_parts[-1]:
        score += 10
    if q_parts and c_parts and q_parts[0] == c_parts[0]:
        score += 10
    elif q_parts and c_parts and q_parts[0][:1] == c_parts[0][:1]:
        score += 4
    return score


def first_last_match(query_name: str, candidate_name: str) -> bool:
    query_parts = normalize_name(query_name).replace("-", " ").split()
    candidate_parts = normalize_name(candidate_name).replace("-", " ").split()
    if len(query_parts) < 2 or len(candidate_parts) < 2:
        return False
    return query_parts[0] == candidate_parts[0] and query_parts[-1] == candidate_parts[-1]


def initial_last_match(query_name: str, candidate_name: str) -> bool:
    query_parts = normalize_name(query_name).replace("-", " ").split()
    candidate_parts = normalize_name(candidate_name).replace("-", " ").split()
    if len(query_parts) < 2 or len(candidate_parts) < 2:
        return False
    return query_parts[0][:1] == candidate_parts[0][:1] and query_parts[-1] == candidate_parts[-1]


def author_score(researcher: dict[str, str], candidate: dict[str, Any], glasgow_id: str) -> float:
    score = name_score(researcher["name"], candidate.get("display_name", ""))
    if has_glasgow_affiliation(candidate, glasgow_id):
        score += 30
    if candidate.get("orcid"):
        score += 3
    score += min(float(candidate.get("works_count") or 0), 200.0) / 100.0
    return score


def search_author_candidates(name: str) -> list[dict[str, Any]]:
    params = {
        "search": name,
        "per-page": AUTHOR_CANDIDATES,
        "select": "id,display_name,orcid,works_count,cited_by_count,last_known_institutions,affiliations",
    }
    data = fetch_json("/authors", params)
    time.sleep(REQUEST_SLEEP)
    return data.get("results", [])


def resolve_author(researcher: dict[str, str], glasgow_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = search_author_candidates(researcher["name"])
    ranked = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate = dict(candidate)
        candidate["_rank"] = rank
        candidate["_score"] = round(author_score(researcher, candidate, glasgow_id), 3)
        candidate["_has_glasgow"] = has_glasgow_affiliation(candidate, glasgow_id)
        ranked.append(candidate)
    ranked.sort(key=lambda c: c["_score"], reverse=True)

    if not ranked:
        return None, ranked

    best = ranked[0]
    glasgow_ranked = [candidate for candidate in ranked if candidate["_has_glasgow"]]
    best_glasgow = glasgow_ranked[0] if glasgow_ranked else None
    second_score = ranked[1]["_score"] if len(ranked) > 1 else 0
    display_ratio = SequenceMatcher(
        None,
        normalize_name(researcher["name"]),
        normalize_name(best.get("display_name", "")),
    ).ratio()

    selected = None
    if best["_has_glasgow"] and first_last_match(researcher["name"], best.get("display_name", "")):
        selected = best
    elif best["_has_glasgow"] and display_ratio >= 0.82 and best["_score"] - second_score >= 8:
        selected = best
    elif best["_has_glasgow"] and display_ratio >= 0.93:
        selected = best
    elif best["_has_glasgow"] and initial_last_match(researcher["name"], best.get("display_name", "")) and best["_score"] >= 105:
        selected = best
    elif best_glasgow and first_last_match(researcher["name"], best_glasgow.get("display_name", "")) and best_glasgow["_score"] >= 105:
        selected = best_glasgow
    elif best_glasgow and initial_last_match(researcher["name"], best_glasgow.get("display_name", "")) and best_glasgow["_score"] >= 95:
        selected = best_glasgow

    return selected, ranked


def iter_author_works(author_id: str) -> list[dict[str, Any]]:
    works = []
    cursor = "*"
    short_id = short_openalex_id(author_id)
    while cursor:
        data = fetch_json(
            "/works",
            {
                "filter": f"author.id:{short_id}",
                "per-page": WORKS_PER_PAGE,
                "cursor": cursor,
                "select": (
                    "id,doi,ids,display_name,title,publication_year,primary_location,"
                    "authorships,abstract_inverted_index,type,cited_by_count,referenced_works"
                ),
            },
        )
        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(REQUEST_SLEEP)
        if not data.get("results"):
            break
    return works


def fetch_work_by_doi(doi: str) -> dict[str, Any] | None:
    doi = normalize_doi(doi)
    if not doi:
        return None
    try:
        return fetch_json(
            f"/works/doi:{doi}",
            {
                "select": (
                    "id,doi,ids,display_name,title,publication_year,primary_location,"
                    "authorships,abstract_inverted_index,type,cited_by_count,referenced_works"
                ),
            },
        )
    except Exception:
        return None


def profile_dois(url: str) -> set[str]:
    """Extract DOI strings from a Glasgow staff publication page."""
    if not url:
        return set()
    try:
        import requests

        resp = requests.get(url, params={"view": "pubs"}, timeout=45)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return set()
    dois = set()
    for match in re.findall(r"(?:doi:\s*|https?://doi\.org/)(10\.[^\s<>()]+)", text, flags=re.I):
        dois.add(normalize_doi(html.unescape(match)))
    return dois


def source_name(work: dict[str, Any]) -> str:
    source = (work.get("primary_location") or {}).get("source") or {}
    return str(source.get("display_name") or "")


def author_names(work: dict[str, Any]) -> str:
    names = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        display_name = author.get("display_name")
        if display_name:
            names.append(str(display_name))
    return "; ".join(names)


def work_row(work: dict[str, Any]) -> dict[str, Any]:
    ids = work.get("ids") or {}
    openalex_id = short_openalex_id(work.get("id", ""))
    refs = work.get("referenced_works", []) or []
    return {
        "paper_id": openalex_id,
        "openalex_id": openalex_id,
        "pmid": pmid_from_ids(ids),
        "year": work.get("publication_year") or "",
        "journal": source_name(work),
        "title": html.unescape(str(work.get("display_name") or work.get("title") or "")),
        "abstract": abstract_from_inverted_index(work.get("abstract_inverted_index")),
        "mesh_terms": "",
        "doi": clean_doi(work.get("doi") or ids.get("doi")),
        "all_authors": author_names(work),
        "type": work.get("type") or "",
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "referenced_works": "|".join(refs),
    }


def write_author_candidates(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "researcher_name",
        "school",
        "college",
        "selected",
        "candidate_rank",
        "score",
        "has_glasgow_affiliation",
        "openalex_author_id",
        "display_name",
        "orcid",
        "works_count",
        "cited_by_count",
        "institutions",
    ]
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_author_overrides() -> dict[str, dict[str, Any]]:
    """Load optional manual author resolutions keyed by researcher name."""
    if not os.path.exists(OVERRIDES_CSV):
        return {}
    df = pd.read_csv(OVERRIDES_CSV)
    if "researcher_name" not in df.columns or "openalex_author_id" not in df.columns:
        raise ValueError(f"{OVERRIDES_CSV} must contain researcher_name and openalex_author_id columns")
    overrides = {}
    for _, row in df.iterrows():
        name = str(row.get("researcher_name", "")).strip()
        author_id = str(row.get("openalex_author_id", "")).strip()
        if name and author_id and author_id.lower() != "nan":
            override = {
                "author_id": author_id,
                "profile_publications_url": str(row.get("profile_publications_url", "") or "").strip(),
                "doi_allowlist": {
                    normalize_doi(doi)
                    for doi in re.split(r"[|;,\s]+", str(row.get("doi_allowlist", "") or ""))
                    if normalize_doi(doi)
                },
            }
            if override["profile_publications_url"] and override["profile_publications_url"].lower() != "nan":
                override["doi_allowlist"].update(profile_dois(override["profile_publications_url"]))
            overrides[normalize_name(name)] = override
            overrides[normalize_name(clean_researcher_name(name))] = override
    return overrides


def merge_unique_works(works: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {}
    for work in works:
        work_id = short_openalex_id(work.get("id", ""))
        if work_id:
            by_id[work_id] = work
    return list(by_id.values())


def assert_can_write(paths: list[str], overwrite: bool) -> None:
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not overwrite:
        rel = ", ".join(os.path.relpath(p, os.getcwd()) for p in existing)
        raise SystemExit(f"Refusing to overwrite existing files: {rel}. Re-run with --overwrite.")


def scrape(overwrite: bool = False) -> None:
    assert_can_write([OUTPUT_CSV, AUTHORS_CSV, CITATIONS_CSV, CANDIDATES_CSV], overwrite)

    print("Parsing researchers from xlsx...")
    researchers = clean_researchers(parse_researchers(XLSX_FILE))
    print(f"  {len(researchers)} researchers")

    glasgow_id = glasgow_institution_id()
    print(f"  University of Glasgow OpenAlex institution: {glasgow_id}")
    overrides = load_author_overrides()
    if overrides:
        print(f"  Loaded {len(overrides)} manual author overrides from {OVERRIDES_CSV}")

    papers: dict[str, dict[str, Any]] = {}
    author_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for researcher in tqdm(researchers, desc="Authors"):
        override = overrides.get(normalize_name(researcher["name"]))
        doi_allowlist: set[str] = set()
        if override:
            doi_allowlist = set(override.get("doi_allowlist", set()))
            selected = {
                "id": override["author_id"],
                "display_name": researcher["name"],
                "_rank": "override",
                "_score": "override",
                "_has_glasgow": "",
            }
            candidates = [selected]
        else:
            selected, candidates = resolve_author(researcher, glasgow_id)
        selected_id = short_openalex_id(selected["id"]) if selected else ""

        for candidate in candidates:
            candidate_id = short_openalex_id(candidate.get("id", ""))
            candidate_rows.append(
                {
                    "researcher_name": researcher["name"],
                    "school": researcher["school"],
                    "college": researcher["college"],
                    "selected": candidate_id == selected_id,
                    "candidate_rank": candidate.get("_rank", ""),
                    "score": candidate.get("_score", ""),
                    "has_glasgow_affiliation": candidate.get("_has_glasgow", False),
                    "openalex_author_id": candidate_id,
                    "display_name": candidate.get("display_name", ""),
                    "orcid": candidate.get("orcid", ""),
                    "works_count": candidate.get("works_count", ""),
                    "cited_by_count": candidate.get("cited_by_count", ""),
                    "institutions": "; ".join(institution_names(candidate)),
                }
            )

        if not selected:
            unresolved.append(researcher["name"])
            continue

        works = iter_author_works(selected["id"])
        if doi_allowlist:
            doi_works = [fetch_work_by_doi(doi) for doi in sorted(doi_allowlist)]
            works = merge_unique_works(works + [work for work in doi_works if work])
            before_filter = len(works)
            works = [
                work for work in works
                if normalize_doi((work.get("doi") or (work.get("ids") or {}).get("doi"))) in doi_allowlist
            ]
            print(
                f"  DOI allowlist for {researcher['name']}: "
                f"retained {len(works)}/{before_filter} OpenAlex works"
            )
        for work in works:
            row = work_row(work)
            if not row["paper_id"] or not row["title"]:
                continue
            papers[row["paper_id"]] = row
            author_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "pmid": row["pmid"],
                    "openalex_id": row["openalex_id"],
                    "author_name": researcher["name"],
                    "school": researcher["school"],
                    "college": researcher["college"],
                    "openalex_author_id": selected_id,
                }
            )

    write_author_candidates(candidate_rows)

    paper_df = pd.DataFrame(papers.values()).sort_values(["year", "paper_id"], ascending=[False, True])
    paper_df.to_csv(OUTPUT_CSV, index=False)

    authors_df = pd.DataFrame(author_rows).drop_duplicates()
    authors_df.to_csv(AUTHORS_CSV, index=False)

    citations_df = paper_df[["paper_id", "pmid", "openalex_id", "cited_by_count"]].copy()
    citations_df.to_csv(CITATIONS_CSV, index=False)

    print(f"\nDone. Wrote {len(paper_df)} unique works to {OUTPUT_CSV}")
    print(f"Author-paper mappings: {len(authors_df)} rows to {AUTHORS_CSV}")
    print(f"Author-resolution audit: {len(candidate_rows)} rows to {CANDIDATES_CSV}")
    if unresolved:
        print(f"Unresolved researchers ({len(unresolved)}): {', '.join(unresolved)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Glasgow CSV outputs.")
    args = parser.parse_args()
    scrape(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
