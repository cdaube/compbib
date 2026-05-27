# CogNeuro Explorer

Interactive browser of cognitive neuroscience and neuroimaging abstracts (1980-2025).

**Explore the live map here: [https://cdaube.github.io/cogneuro/](https://cdaube.github.io/cogneuro/)**

**Glasgow Research Explorer: [https://cdaube.github.io/cogneuro/glasgow_explorer.html](https://cdaube.github.io/cogneuro/glasgow_explorer.html)**

*These tools use LLM embeddings and UMAP to visualize the semantic landscape of research abstracts. The Glasgow explorer maps ~13k evidence-filtered works from University of Glasgow researchers, coloured by school, college, year, or citation network.*

## Glasgow Scraping Pipeline

The Glasgow explorer starts from `data/MVLS Imaging Initiative List of Academics.xlsx`, which defines the researcher roster, school, and college labels. The scraper keeps this spreadsheet as the roster of people to include; it does not scrape all Glasgow staff automatically.

The main scraper is:

```bash
uv run python scripts/scrape_glasgow_openalex.py --overwrite
```

For each spreadsheet researcher, the scraper searches OpenAlex author records and writes the author-resolution audit to `data/glasgow_author_candidates.csv`. Candidate author records are scored using name similarity, Glasgow affiliation evidence, ORCID presence, and OpenAlex work counts. Manual fixes for ambiguous or fragile people live in `data/glasgow_author_overrides.csv`.

Selecting an OpenAlex author ID does not automatically include all works from that author cluster. OpenAlex author clusters can contain homonyms or merged identities, so the scraper treats every fetched work as a candidate and applies a work-level identity evidence filter. Evidence includes ORCID DOI/title matches, manual DOI allowlists, Glasgow/profile DOI evidence, work-level affiliations, trusted coauthor overlap, plausible career dates from ORCID, PMID evidence, and OpenAlex publication type. Implausible works are excluded; uncertain works are quarantined.

The evidence-filtering outputs are:

- `data/glasgow_candidate_works.csv`: all fetched candidate author-work rows.
- `data/glasgow_work_evidence.csv`: include/quarantine/exclude decision, score, and evidence flags for every candidate row.
- `data/glasgow_rejected_works.csv`: rejected and quarantined candidate rows for inspection.
- `data/glasgow_abstracts.csv`: included works only.
- `data/glasgow_authors.csv`: included researcher-work mappings only.
- `data/glasgow_citations.csv`: citation counts for included works.

After scraping, rebuild the citation graph and explorer data:

```bash
uv run python scripts/enrich_glasgow_citation_graph.py
uv run python scripts/build_glasgow_explorer_data.py --force
```

The final command recomputes embeddings, the 10 UMAP projections, and `glasgow_explorer.html`.
