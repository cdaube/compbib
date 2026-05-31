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
  data/glasgow_candidate_works.csv    all fetched author/work candidates
  data/glasgow_work_evidence.csv      per-candidate include/quarantine/exclude evidence
  data/glasgow_rejected_works.csv     rejected or quarantined candidates

Usage:
    uv run python scripts/scrape_glasgow_openalex.py --overwrite

Optional:
  data/glasgow_author_overrides.csv with researcher_name,openalex_author_id
  can be used to resolve ambiguous people from glasgow_author_candidates.csv.
  Optional columns profile_publications_url and doi_allowlist add positive
  evidence for fragile/common-name author records. doi_allowlist is treated as
  hard include evidence; Glasgow profile publications are not used as a
  complete ground truth.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
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
CANDIDATE_WORKS_CSV = os.path.join(DATA_DIR, "glasgow_candidate_works.csv")
WORK_EVIDENCE_CSV = os.path.join(DATA_DIR, "glasgow_work_evidence.csv")
REJECTED_WORKS_CSV = os.path.join(DATA_DIR, "glasgow_rejected_works.csv")
ORCID_CACHE = os.path.join(DATA_DIR, ".glasgow_orcid_cache.json")

OPENALEX_BASE = "https://api.openalex.org"
ORCID_BASE = "https://pub.orcid.org/v3.0"
POLITE_EMAIL = "christoph.daube@gmail.com"
GLASGOW_ROR = "https://ror.org/00vtgdb53"

REQUEST_SLEEP = 0.25
MAX_RETRIES = 8
WORKS_PER_PAGE = 200
AUTHOR_CANDIDATES = 10

TRUSTED_DOI_INCLUDE_SCORE = 70
WORK_AFFILIATION_INCLUDE_SCORE = 45
COAUTHOR_INCLUDE_SCORE = 35
QUARANTINE_SCORE = 20
CAREER_START_GRACE_YEARS = 3
ORCID_EARLIEST_GRACE_YEARS = 12
# Per-researcher date-sanity floor: a work whose year sits far below the
# researcher's own body of work (and lacks any Glasgow affiliation) is almost
# always an OpenAlex cluster merge of an older homonym. Drop those isolated
# old outliers even when other evidence (incl. trusted DOIs) would include them.
DATE_FLOOR_GRACE_YEARS = 15
DATE_FLOOR_MIN_WORKS = 8
# A below-floor work is normally kept if it carries any Glasgow affiliation (it
# may be genuine early-career output). But a Glasgow-affiliated work separated
# from the rest of the researcher's record by this many years is almost always
# a different, earlier same-name Glasgow person merged in by OpenAlex.
DATE_FLOOR_ISOLATION_GAP_YEARS = 25
NON_PUBLICATION_TYPES = {
    "dataset",
    "peer-review",
    "supplementary-materials",
    "erratum",
    "reference-entry",
    "other",
}


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
    if not re.match(r"^10\.\d{4,9}/\S+$", doi, flags=re.I):
        return ""
    return doi.lower()


def normalize_orcid(value: str | None) -> str:
    if not value:
        return ""
    match = re.search(r"(\d{4}-\d{4}-\d{4}-[\dX]{4})", str(value), flags=re.I)
    return match.group(1).upper() if match else ""


def compact_title(value: str | None) -> str:
    value = html.unescape(str(value or "")).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
    except TypeError:
        if value is None:
            return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"\d{4}", text)
    return int(match.group(0)) if match else None


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
        except requests.HTTPError as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                raise
            if status == 429:
                retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                try:
                    wait = float(retry_after) if retry_after else 0
                except ValueError:
                    wait = 0
                wait = max(wait, 30 * (attempt + 1))
                print(f"OpenAlex 429; waiting {wait:.0f}s before retrying {path}", flush=True)
                time.sleep(wait)
            else:
                time.sleep(2 ** (attempt + 1))
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_openalex_author(author_id: str) -> dict[str, Any] | None:
    short_id = short_openalex_id(author_id)
    if not short_id:
        return None
    try:
        return fetch_json(
            f"/authors/{short_id}",
            {
                "select": (
                    "id,display_name,orcid,works_count,cited_by_count,"
                    "last_known_institutions,affiliations"
                ),
            },
        )
    except Exception:
        return None


def load_orcid_cache() -> dict[str, Any]:
    if not os.path.exists(ORCID_CACHE):
        return {}
    try:
        with open(ORCID_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_orcid_cache(cache: dict[str, Any]) -> None:
    with open(ORCID_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


def fetch_orcid_record(orcid: str, cache: dict[str, Any]) -> dict[str, Any]:
    """Fetch a public ORCID record summary, cached between scraper runs."""
    orcid = normalize_orcid(orcid)
    if not orcid:
        return {}
    if orcid in cache:
        return cache[orcid] or {}
    try:
        resp = requests.get(
            f"{ORCID_BASE}/{orcid}/record",
            headers={"Accept": "application/json"},
            timeout=45,
        )
        resp.raise_for_status()
        cache[orcid] = resp.json()
    except Exception:
        cache[orcid] = {}
    time.sleep(REQUEST_SLEEP)
    return cache[orcid] or {}


def nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def orcid_work_summaries(record: dict[str, Any]) -> list[dict[str, Any]]:
    works = nested_value(record, "activities-summary", "works", "group") or []
    summaries = []
    for group in works:
        summaries.extend(group.get("work-summary") or [])
    return summaries


def orcid_external_dois(summary: dict[str, Any]) -> set[str]:
    dois = set()
    external_ids = nested_value(summary, "external-ids", "external-id") or []
    for ext_id in external_ids:
        ext_type = str(ext_id.get("external-id-type") or "").lower()
        ext_value = str(ext_id.get("external-id-value") or "")
        if ext_type == "doi":
            doi = normalize_doi(ext_value)
            if doi:
                dois.add(doi)
    return dois


def orcid_summary_title(summary: dict[str, Any]) -> str:
    return str(nested_value(summary, "title", "title", "value") or "")


def orcid_summary_year(summary: dict[str, Any]) -> int | None:
    return safe_int(nested_value(summary, "publication-date", "year", "value"))


def orcid_affiliation_summaries(record: dict[str, Any], section: str) -> list[dict[str, Any]]:
    groups = nested_value(record, "activities-summary", section, "affiliation-group") or []
    summaries = []
    for group in groups:
        for item in group.get("summaries") or []:
            if isinstance(item, dict):
                summaries.extend(value for value in item.values() if isinstance(value, dict))
    return summaries


def orcid_affiliation_names(record: dict[str, Any]) -> set[str]:
    names = set()
    for section in ("employments", "educations", "qualifications"):
        for summary in orcid_affiliation_summaries(record, section):
            name = nested_value(summary, "organization", "name")
            if name:
                names.add(normalize_name(str(name)))
    return names


def orcid_career_start_year(record: dict[str, Any]) -> int | None:
    years = []
    for section in ("employments", "educations", "qualifications"):
        for summary in orcid_affiliation_summaries(record, section):
            year = safe_int(nested_value(summary, "start-date", "year", "value"))
            if year:
                years.append(year)
    return min(years) if years else None


def orcid_profile(orcid: str, cache: dict[str, Any]) -> dict[str, Any]:
    record = fetch_orcid_record(orcid, cache)
    summaries = orcid_work_summaries(record)
    dois: set[str] = set()
    titles: set[str] = set()
    years: list[int] = []
    for summary in summaries:
        dois.update(orcid_external_dois(summary))
        title = compact_title(orcid_summary_title(summary))
        if title:
            titles.add(title)
        year = orcid_summary_year(summary)
        if year:
            years.append(year)
    return {
        "orcid": normalize_orcid(orcid),
        "work_dois": dois,
        "work_titles": titles,
        "work_years": years,
        "earliest_work_year": min(years) if years else None,
        "career_start_year": orcid_career_start_year(record),
        "affiliation_names": orcid_affiliation_names(record),
        "works_count": len(summaries),
    }


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


def core_name_tokens(name: str) -> list[str]:
    """Name tokens with initials dropped (e.g. 'Andrew K. Reilly' -> ['andrew','reilly'])."""
    parts = normalize_name(name).replace("-", " ").split()
    return [p for p in parts if len(p) > 1]


def same_core_name(query_name: str, candidate_name: str) -> bool:
    """True when two names share the same non-initial token sequence.

    Blocks compound-surname homonym merges such as 'Andrew Reilly' vs
    'Andrew Luxton-Reilly' that first_last_match would otherwise treat as equal.
    """
    return bool(core_name_tokens(query_name)) and core_name_tokens(query_name) == core_name_tokens(candidate_name)


def author_score(researcher: dict[str, str], candidate: dict[str, Any], glasgow_id: str) -> float:
    score = name_score(researcher["name"], candidate.get("display_name", ""))
    if has_glasgow_affiliation(candidate, glasgow_id):
        score += 30
    if candidate.get("orcid"):
        score += 3
    score += min(float(candidate.get("works_count") or 0), 200.0) / 100.0
    return score


def author_count(candidate: dict[str, Any], key: str) -> int:
    try:
        return int(candidate.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def prefer_orcid_backed_duplicate(
    researcher: dict[str, str],
    selected: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Prefer a rich ORCID-backed author over tiny exact-name OpenAlex fragments."""
    if not selected:
        return selected

    selected_works = author_count(selected, "works_count")
    selected_citations = author_count(selected, "cited_by_count")
    if selected.get("orcid") and selected_works >= 25:
        return selected
    if selected_works > 25:
        return selected

    rich_duplicates = []
    for candidate in ranked:
        if candidate is selected:
            continue
        if not candidate.get("_has_glasgow") or not candidate.get("orcid"):
            continue
        if not first_last_match(researcher["name"], candidate.get("display_name", "")):
            continue
        # Require an exact core-name match before merging into a larger cluster,
        # so a richer homonym (e.g. 'Andrew Luxton-Reilly') is not mistaken for
        # the roster person ('Andrew Reilly').
        if not same_core_name(researcher["name"], candidate.get("display_name", "")):
            continue
        works = author_count(candidate, "works_count")
        citations = author_count(candidate, "cited_by_count")
        if works < max(50, selected_works * 5):
            continue
        if citations < max(500, selected_citations * 5):
            continue
        rich_duplicates.append(candidate)

    if not rich_duplicates:
        return selected
    return max(
        rich_duplicates,
        key=lambda candidate: (
            author_count(candidate, "works_count"),
            author_count(candidate, "cited_by_count"),
            candidate.get("_score", 0),
        ),
    )


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

    selected = prefer_orcid_backed_duplicate(researcher, selected, ranked)
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


def work_title(work: dict[str, Any]) -> str:
    return html.unescape(str(work.get("display_name") or work.get("title") or ""))


def work_doi(work: dict[str, Any]) -> str:
    ids = work.get("ids") or {}
    return normalize_doi(work.get("doi") or ids.get("doi"))


def work_year(work: dict[str, Any]) -> int | None:
    return safe_int(work.get("publication_year"))


def work_pmid(work: dict[str, Any]) -> str:
    return pmid_from_ids(work.get("ids") or {})


def authorships(work: dict[str, Any]) -> list[dict[str, Any]]:
    return list(work.get("authorships") or [])


def authorship_author_id(authorship: dict[str, Any]) -> str:
    return short_openalex_id((authorship.get("author") or {}).get("id", ""))


def authorship_author_name(authorship: dict[str, Any]) -> str:
    return str((authorship.get("author") or {}).get("display_name") or "")


def selected_authorships(work: dict[str, Any], selected_author_id: str, researcher_name: str) -> list[dict[str, Any]]:
    selected_id = short_openalex_id(selected_author_id)
    matched = [
        authorship for authorship in authorships(work)
        if selected_id and authorship_author_id(authorship) == selected_id
    ]
    if matched:
        return matched
    return [
        authorship for authorship in authorships(work)
        if initial_last_match(researcher_name, authorship_author_name(authorship))
    ]


def authorship_institution_names(authorship: dict[str, Any]) -> list[str]:
    names = []
    for inst in authorship.get("institutions") or []:
        names.append(str(inst.get("display_name", "")))
        names.append(str(inst.get("ror", "")))
        names.append(short_openalex_id(inst.get("id", "")))
    for raw in authorship.get("raw_affiliation_strings") or []:
        names.append(str(raw))
    return [name for name in names if name]


def work_has_glasgow_author_affiliation(
    work: dict[str, Any],
    selected_author_id: str,
    researcher_name: str,
    glasgow_id: str,
) -> bool:
    for authorship in selected_authorships(work, selected_author_id, researcher_name):
        names = authorship_institution_names(authorship)
        joined = " | ".join(names).lower()
        if glasgow_id in names or GLASGOW_ROR.lower() in joined or "university of glasgow" in joined:
            return True
    return False


def work_has_any_glasgow_affiliation(work: dict[str, Any], glasgow_id: str) -> bool:
    for authorship in authorships(work):
        names = authorship_institution_names(authorship)
        joined = " | ".join(names).lower()
        if glasgow_id in names or GLASGOW_ROR.lower() in joined or "university of glasgow" in joined:
            return True
    return False


def work_has_known_orcid_affiliation(
    work: dict[str, Any],
    selected_author_id: str,
    researcher_name: str,
    known_affiliations: set[str],
) -> bool:
    if not known_affiliations:
        return False
    for authorship in selected_authorships(work, selected_author_id, researcher_name):
        names = {normalize_name(name) for name in authorship_institution_names(authorship)}
        for known in known_affiliations:
            if known in names or any(known and known in name for name in names):
                return True
    return False


def normalized_coauthors(work: dict[str, Any], researcher_name: str) -> set[str]:
    names = set()
    researcher = normalize_name(researcher_name)
    for authorship in authorships(work):
        name = normalize_name(authorship_author_name(authorship))
        if name and name != researcher:
            names.add(name)
    return names


def build_trusted_coauthors(
    works: list[dict[str, Any]],
    researcher_name: str,
    trusted_dois: set[str],
    trusted_titles: set[str],
) -> set[str]:
    coauthors: set[str] = set()
    for work in works:
        doi = work_doi(work)
        title = compact_title(work_title(work))
        if (doi and doi in trusted_dois) or (title and title in trusted_titles):
            coauthors.update(normalized_coauthors(work, researcher_name))
    return coauthors


def score_work_identity(
    *,
    work: dict[str, Any],
    researcher: dict[str, str],
    selected_author_id: str,
    selected_orcid: str,
    glasgow_id: str,
    explicit_dois: set[str],
    profile_dois_set: set[str],
    orcid_info: dict[str, Any],
    trusted_coauthors: set[str],
) -> dict[str, Any]:
    doi = work_doi(work)
    title = compact_title(work_title(work))
    year = work_year(work)
    work_type = str(work.get("type") or "").strip().lower()
    flags: list[str] = []
    score = 0

    orcid_dois = set(orcid_info.get("work_dois") or set())
    orcid_titles = set(orcid_info.get("work_titles") or set())
    career_start = orcid_info.get("career_start_year")
    earliest_orcid_work = orcid_info.get("earliest_work_year")
    known_orcid_affiliations = set(orcid_info.get("affiliation_names") or set())

    trusted_doi = False
    if doi and doi in explicit_dois:
        score += 100
        trusted_doi = True
        flags.append("manual_doi_allowlist")
    if doi and doi in orcid_dois:
        score += TRUSTED_DOI_INCLUDE_SCORE
        trusted_doi = True
        flags.append("orcid_doi")
    if title and title in orcid_titles:
        score += TRUSTED_DOI_INCLUDE_SCORE
        trusted_doi = True
        flags.append("orcid_title")
    if doi and doi in profile_dois_set:
        score += 55
        trusted_doi = True
        flags.append("profile_doi")

    selected_name_match = any(
        initial_last_match(researcher["name"], authorship_author_name(authorship))
        for authorship in selected_authorships(work, selected_author_id, researcher["name"])
    )
    if selected_name_match:
        score += 8
        flags.append("selected_author_name_match")

    if work_has_glasgow_author_affiliation(work, selected_author_id, researcher["name"], glasgow_id):
        score += WORK_AFFILIATION_INCLUDE_SCORE
        flags.append("selected_author_glasgow_affiliation")
    elif work_has_known_orcid_affiliation(
        work,
        selected_author_id,
        researcher["name"],
        known_orcid_affiliations,
    ):
        score += 30
        flags.append("selected_author_known_orcid_affiliation")
    elif work_has_any_glasgow_affiliation(work, glasgow_id):
        score += 18
        flags.append("some_author_glasgow_affiliation")

    pmid = work_pmid(work)
    if pmid:
        score += 8
        flags.append("has_pmid")
        if work_has_any_glasgow_affiliation(work, glasgow_id):
            score += 15
            flags.append("pubmed_like_with_glasgow_affiliation")

    coauthors = normalized_coauthors(work, researcher["name"])
    overlap = coauthors & trusted_coauthors
    if len(overlap) >= 2:
        score += COAUTHOR_INCLUDE_SCORE
        flags.append(f"trusted_coauthor_overlap:{len(overlap)}")
    elif len(overlap) == 1:
        score += 15
        flags.append("trusted_coauthor_overlap:1")

    if selected_orcid:
        flags.append("selected_author_has_orcid")
        score += 2

    hard_exclude = False
    if year and career_start and year < career_start - CAREER_START_GRACE_YEARS and not trusted_doi:
        hard_exclude = True
        score -= 100
        flags.append(f"predates_orcid_career_start:{career_start}")
    elif year and earliest_orcid_work and year < earliest_orcid_work - ORCID_EARLIEST_GRACE_YEARS and not trusted_doi:
        hard_exclude = True
        score -= 75
        flags.append(f"predates_orcid_work_span:{earliest_orcid_work}")

    if not doi and not pmid and not trusted_doi:
        score -= 10
        flags.append("no_doi_or_pmid")

    if work_type in NON_PUBLICATION_TYPES and not (doi and doi in explicit_dois):
        decision = "exclude"
        score -= 60
        flags.append(f"non_publication_type:{work_type}")
    elif trusted_doi:
        decision = "include"
    elif hard_exclude:
        decision = "exclude"
    elif score >= WORK_AFFILIATION_INCLUDE_SCORE:
        decision = "include"
    elif score >= QUARANTINE_SCORE:
        decision = "quarantine"
    else:
        decision = "exclude"

    return {
        "researcher_name": researcher["name"],
        "school": researcher["school"],
        "college": researcher["college"],
        "paper_id": short_openalex_id(work.get("id", "")),
        "openalex_id": short_openalex_id(work.get("id", "")),
        "pmid": pmid,
        "doi": clean_doi(work.get("doi") or (work.get("ids") or {}).get("doi")),
        "year": year or "",
        "type": work_type,
        "title": work_title(work),
        "journal": source_name(work),
        "all_authors": author_names(work),
        "openalex_author_id": short_openalex_id(selected_author_id),
        "orcid": selected_orcid,
        "identity_decision": decision,
        "identity_score": round(score, 3),
        "identity_flags": "|".join(flags),
    }


def _percentile(values: list[int], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 1]); no numpy dependency."""
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def apply_date_floor(scored: list[dict[str, Any]]) -> int:
    """Demote isolated, very-old included works to 'exclude'.

    A work is dropped if its year sits more than DATE_FLOOR_GRACE_YEARS below the
    researcher's 10th-percentile publication year and either (a) it carries no
    Glasgow affiliation evidence, or (b) it is separated from the rest of the
    researcher's record by at least DATE_FLOOR_ISOLATION_GAP_YEARS. Mutates the
    evidence dicts in place and returns the number of works demoted.
    """
    included = [e for e in scored if e["identity_decision"] == "include"]
    years = [int(e["year"]) for e in included if str(e.get("year", "")).strip().isdigit()]
    if len(years) < DATE_FLOOR_MIN_WORKS:
        return 0
    p10 = _percentile(years, 0.10)
    if p10 is None:
        return 0
    floor_year = p10 - DATE_FLOOR_GRACE_YEARS
    sorted_years = sorted(years)
    demoted = 0
    for e in included:
        year_text = str(e.get("year", "")).strip()
        if not year_text.isdigit():
            continue
        year = int(year_text)
        if year >= floor_year:
            continue
        if "glasgow_affiliation" in (e.get("identity_flags") or ""):
            higher = [y for y in sorted_years if y > year]
            gap = (min(higher) - year) if higher else 0
            if gap < DATE_FLOOR_ISOLATION_GAP_YEARS:
                continue
        e["identity_decision"] = "exclude"
        flags = e.get("identity_flags") or ""
        e["identity_flags"] = (flags + "|" if flags else "") + f"date_outlier_below_floor:{int(floor_year)}"
        demoted += 1
    return demoted


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


def write_dict_rows(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
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
                "exclude": author_id.strip().upper() == "EXCLUDE",
                "profile_publications_url": str(row.get("profile_publications_url", "") or "").strip(),
                "explicit_doi_allowlist": {
                    normalize_doi(doi)
                    for doi in re.split(r"[|;,\s]+", str(row.get("doi_allowlist", "") or ""))
                    if normalize_doi(doi)
                },
                "profile_dois": set(),
            }
            if override["profile_publications_url"] and override["profile_publications_url"].lower() != "nan":
                override["profile_dois"].update(profile_dois(override["profile_publications_url"]))
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
    assert_can_write(
        [
            OUTPUT_CSV,
            AUTHORS_CSV,
            CITATIONS_CSV,
            CANDIDATES_CSV,
            CANDIDATE_WORKS_CSV,
            WORK_EVIDENCE_CSV,
            REJECTED_WORKS_CSV,
        ],
        overwrite,
    )

    print("Parsing researchers from xlsx...")
    researchers = clean_researchers(parse_researchers(XLSX_FILE))
    print(f"  {len(researchers)} researchers")

    glasgow_id = glasgow_institution_id()
    print(f"  University of Glasgow OpenAlex institution: {glasgow_id}")
    overrides = load_author_overrides()
    if overrides:
        print(f"  Loaded {len(overrides)} manual author overrides from {OVERRIDES_CSV}")
    orcid_cache = load_orcid_cache()

    papers: dict[str, dict[str, Any]] = {}
    author_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_work_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for researcher in tqdm(researchers, desc="Authors"):
        override = overrides.get(normalize_name(researcher["name"]))
        if override and override.get("exclude"):
            candidate_rows.append(
                {
                    "researcher_name": researcher["name"],
                    "school": researcher["school"],
                    "college": researcher["college"],
                    "selected": False,
                    "candidate_rank": "override",
                    "score": "override",
                    "has_glasgow_affiliation": "",
                    "openalex_author_id": "EXCLUDE",
                    "display_name": "(excluded by override)",
                    "orcid": "",
                    "works_count": "",
                    "cited_by_count": "",
                    "institutions": "",
                }
            )
            unresolved.append(f"{researcher['name']} (excluded by override)")
            continue
        explicit_dois: set[str] = set()
        profile_dois_set: set[str] = set()
        if override:
            explicit_dois = set(override.get("explicit_doi_allowlist", set()))
            profile_dois_set = set(override.get("profile_dois", set()))
            selected = fetch_openalex_author(override["author_id"]) or {
                "id": override["author_id"],
                "display_name": researcher["name"],
                "orcid": "",
                "_rank": "override",
                "_score": "override",
                "_has_glasgow": "",
            }
            selected = dict(selected)
            selected["_rank"] = "override"
            selected["_score"] = "override"
            selected["_has_glasgow"] = has_glasgow_affiliation(selected, glasgow_id)
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

        selected_orcid = normalize_orcid(selected.get("orcid"))
        orcid_info = orcid_profile(selected_orcid, orcid_cache) if selected_orcid else {}
        works = iter_author_works(selected["id"])
        trusted_dois = explicit_dois | profile_dois_set | set(orcid_info.get("work_dois") or set())
        rescue_dois = explicit_dois | profile_dois_set
        doi_works = [fetch_work_by_doi(doi) for doi in sorted(rescue_dois)]
        works = merge_unique_works(works + [work for work in doi_works if work])

        trusted_coauthors = build_trusted_coauthors(
            works,
            researcher["name"],
            trusted_dois,
            set(orcid_info.get("work_titles") or set()),
        )
        # Pass 1: score every candidate work for this researcher.
        scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for work in works:
            row = work_row(work)
            if not row["paper_id"] or not row["title"]:
                continue
            evidence = score_work_identity(
                work=work,
                researcher=researcher,
                selected_author_id=selected_id,
                selected_orcid=selected_orcid,
                glasgow_id=glasgow_id,
                explicit_dois=explicit_dois,
                profile_dois_set=profile_dois_set,
                orcid_info=orcid_info,
                trusted_coauthors=trusted_coauthors,
            )
            scored.append((evidence, row))

        # Pass 2: drop isolated, implausibly-old works merged from a homonym.
        demoted = apply_date_floor([evidence for evidence, _ in scored])

        # Pass 3: commit decisions and record the audit rows.
        kept = quarantined = excluded = 0
        for evidence, row in scored:
            candidate_work_rows.append(evidence)
            evidence_rows.append(evidence)
            if evidence["identity_decision"] != "include":
                rejected_rows.append(evidence)
                if evidence["identity_decision"] == "quarantine":
                    quarantined += 1
                else:
                    excluded += 1
                continue
            kept += 1
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
        if quarantined or excluded:
            tqdm.write(
                f"  Identity filter for {researcher['name']}: "
                f"included {kept}, quarantined {quarantined}, excluded {excluded}"
                + (f" ({demoted} old-outlier)" if demoted else "")
            )

    write_author_candidates(candidate_rows)
    save_orcid_cache(orcid_cache)

    evidence_fieldnames = [
        "researcher_name",
        "school",
        "college",
        "paper_id",
        "openalex_id",
        "pmid",
        "doi",
        "year",
        "type",
        "title",
        "journal",
        "all_authors",
        "openalex_author_id",
        "orcid",
        "identity_decision",
        "identity_score",
        "identity_flags",
    ]
    write_dict_rows(CANDIDATE_WORKS_CSV, candidate_work_rows, evidence_fieldnames)
    write_dict_rows(WORK_EVIDENCE_CSV, evidence_rows, evidence_fieldnames)
    write_dict_rows(REJECTED_WORKS_CSV, rejected_rows, evidence_fieldnames)

    paper_df = pd.DataFrame(papers.values()).sort_values(["year", "paper_id"], ascending=[False, True])
    paper_df.to_csv(OUTPUT_CSV, index=False)

    authors_df = pd.DataFrame(author_rows).drop_duplicates()
    authors_df.to_csv(AUTHORS_CSV, index=False)

    citations_df = paper_df[["paper_id", "pmid", "openalex_id", "cited_by_count"]].copy()
    citations_df.to_csv(CITATIONS_CSV, index=False)

    print(f"\nDone. Wrote {len(paper_df)} unique works to {OUTPUT_CSV}")
    print(f"Author-paper mappings: {len(authors_df)} rows to {AUTHORS_CSV}")
    print(f"Author-resolution audit: {len(candidate_rows)} rows to {CANDIDATES_CSV}")
    print(f"Candidate work audit: {len(candidate_work_rows)} rows to {CANDIDATE_WORKS_CSV}")
    print(f"Identity evidence audit: {len(evidence_rows)} rows to {WORK_EVIDENCE_CSV}")
    print(f"Rejected/quarantined work audit: {len(rejected_rows)} rows to {REJECTED_WORKS_CSV}")
    if unresolved:
        print(f"Unresolved researchers ({len(unresolved)}): {', '.join(unresolved)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Glasgow CSV outputs.")
    args = parser.parse_args()
    scrape(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
