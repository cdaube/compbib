"""
Build an interactive Glasgow UMAP explorer as a single self-contained HTML file.

Features:
- Scatter plot of Glasgow researcher abstracts in UMAP space
- Dropdown to switch colour mapping: School, College, Year, Citation network
- Collapsible side panel with full abstract + metadata (coloured by school)
- Citation edge overlays (cites → blue, cited-by → red)
- GitHub-Pages-friendly: one HTML file, Plotly loaded from CDN

Usage:
    uv run python scripts/make_glasgow_explorer.py
"""

import json
import os
import colorsys

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = BASE_DIR  # put index-glasgow.html at repo root for GitHub Pages
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Distinguishable colours (port of T. E. Holy's MATLAB function)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb):
    lin = _srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.column_stack([L, a, b])


def distinguishable_colors(n_colors, bg=None):
    if bg is None:
        bg = np.array([[1.0, 1.0, 1.0]])
    bg = np.atleast_2d(bg).astype(float)

    n_grid = 30
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x, indexing="ij")
    rgb = np.column_stack([R.ravel(), G.ravel(), B.ravel()])

    lab = _rgb_to_lab(rgb)
    bglab = _rgb_to_lab(bg)

    mindist2 = np.full(len(rgb), np.inf)
    for i in range(len(bglab) - 1):
        d = np.sum((lab - bglab[i]) ** 2, axis=1)
        mindist2 = np.minimum(d, mindist2)

    colors = np.zeros((n_colors, 3))
    lastlab = bglab[-1]
    for i in range(n_colors):
        d = np.sum((lab - lastlab) ** 2, axis=1)
        mindist2 = np.minimum(d, mindist2)
        idx = np.argmax(mindist2)
        colors[i] = rgb[idx]
        lastlab = lab[idx]

    return colors


# ---------------------------------------------------------------------------

COLLEGE_COLORS = {
    "MVLS": "#2563eb",
    "CoSE": "#dc2626",
    "NHS": "#16a34a",
    "Arts & Humanities": "#a855f7",
}

SCHOOL_ORDER = [
    "School of Biodiversity, One Health & Veterinary Medicine",
    "School of Cancer Sciences",
    "School of Cardiovascular & Metabolic Health",
    "School of Health & Wellbeing",
    "School of Infection & Immunity",
    "School of Medicine, Dentistry & Nursing",
    "School of Molecular Biosciences",
    "School of Psychology & Neuroscience",
    "School of Mathematics and Statistics",
    "School of Physics and Astronomy",
    "School of Computing Science",
    "School of Chemistry",
    "James Watt School of Engineering",
    "School of Geographical and earth Sciences",
    "School of Biomedical Engineering",
]

MVLS_SCHOOLS = [
    "School of Biodiversity, One Health & Veterinary Medicine",
    "School of Cancer Sciences",
    "School of Cardiovascular & Metabolic Health",
    "School of Health & Wellbeing",
    "School of Infection & Immunity",
    "School of Medicine, Dentistry & Nursing",
    "School of Molecular Biosciences",
    "School of Psychology & Neuroscience",
]

BLUE_HUE_SCHOOLS = [
    "School of Mathematics and Statistics",
    "School of Physics and Astronomy",
    "School of Computing Science",
    "School of Chemistry",
    "James Watt School of Engineering",
    "School of Geographical and earth Sciences",
    "School of Biomedical Engineering",
]

SCHOOL_ORDER_INDEX = {school: idx for idx, school in enumerate(SCHOOL_ORDER)}
SCHOOL_NORMALIZATION = {
    "smdn": "School of Medicine, Dentistry & Nursing",
    "school of medicine, dentistry & nursing": "School of Medicine, Dentistry & Nursing",
    "school of medicine, dentistry and nursing": "School of Medicine, Dentistry & Nursing",
    "school of infection & immunology": "School of Infection & Immunity",
    "school of infection and immunology": "School of Infection & Immunity",
    "school of infection & immunity": "School of Infection & Immunity",
    "school of infection and immunity": "School of Infection & Immunity",
    "james watt school of engineering": "James Watt School of Engineering",
    "suerc": "James Watt School of Engineering",
    "school of geographical and earth sciences": "School of Geographical and earth Sciences",
    "school of humanities": "",
}

LEGACY_SCHOOL_COLORS = {
    "James Watt School of Engineering": "rgb(0,0,255)",
    "SMDN": "rgb(0,255,0)",
    "SUERC": "rgb(255,0,0)",
    "School of Biodiversity, One Health & Veterinary Medicine": "rgb(255,0,175)",
    "School of Biomedical Engineering": "rgb(255,211,8)",
    "School of Cancer Sciences": "rgb(0,131,246)",
    "School of Cardiovascular & Metabolic Health": "rgb(0,140,70)",
    "School of Chemistry": "rgb(175,105,61)",
    "School of Computing Science": "rgb(87,8,96)",
    "School of Geographical and earth Sciences": "rgb(0,140,167)",
    "School of Health & Wellbeing": "rgb(255,175,255)",
    "School of Humanities": "rgb(0,255,237)",
    "School of Infection & Immunology": "rgb(193,255,123)",
    "School of Mathematics and Statistics": "rgb(175,87,246)",
    "School of Medicine, Dentistry & Nursing": "rgb(202,0,70)",
    "School of Molecular Biosciences": "rgb(131,140,0)",
    "School of Physics and Astronomy": "rgb(140,105,131)",
    "School of Psychology & Neuroscience": "rgb(43,61,0)",
}

# Additional blue-ish hues selected from distinguishable_colors(100),
# reusing the existing four blue family colours requested by the user.
BLUE_PALETTE_EXTRA_INDICES = (35, 51, 94)


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_identifier(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def clean_year(value):
    if pd.isna(value):
        return ""
    year = pd.to_numeric(value, errors="coerce")
    if pd.notna(year):
        return str(int(year))
    return clean(value)


def normalize_school(value):
    cleaned = clean(value)
    if not cleaned:
        return ""
    return SCHOOL_NORMALIZATION.get(cleaned.lower(), cleaned)


def school_sort_key(value):
    return (SCHOOL_ORDER_INDEX.get(value, len(SCHOOL_ORDER)), value)


def rgb_to_css(rgb):
    return f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"


def perceptual_palette_sort_key(rgb):
    hue, sat, _ = colorsys.rgb_to_hsv(*rgb.tolist())
    brightness = float(_rgb_to_lab(np.array([rgb]))[0, 0])
    chroma = float(np.max(rgb) - np.min(rgb))

    if sat < 0.12 or chroma < 0.08:
        return (11, brightness, 0.0, -sat)

    deg = hue * 360
    if deg < 15 or deg >= 345:
        bucket = 0   # red
    elif deg < 45:
        bucket = 1   # orange
    elif deg < 75:
        bucket = 2   # yellow
    elif deg < 105:
        bucket = 3   # yellow-green
    elif deg < 150:
        bucket = 4   # green
    elif deg < 185:
        bucket = 5   # teal
    elif deg < 210:
        bucket = 6   # cyan
    elif deg < 250:
        bucket = 7   # blue
    elif deg < 280:
        bucket = 8   # indigo
    elif deg < 315:
        bucket = 9   # violet
    else:
        bucket = 10  # magenta / pink

    return (bucket, brightness, deg, -sat)


def build_picker_palette():
    palette = distinguishable_colors(100, bg=np.array([[1, 1, 1], [0, 0, 0]]))
    ordered = sorted(palette, key=perceptual_palette_sort_key)
    return [rgb_to_css(rgb) for rgb in ordered]


def build_school_color_map():
    palette_100 = distinguishable_colors(100, bg=np.array([[1, 1, 1], [0, 0, 0]]))
    extra_blue_hues = [rgb_to_css(palette_100[idx]) for idx in BLUE_PALETTE_EXTRA_INDICES]
    mvls_hues = [
        LEGACY_SCHOOL_COLORS["School of Biodiversity, One Health & Veterinary Medicine"],
        LEGACY_SCHOOL_COLORS["SUERC"],
        LEGACY_SCHOOL_COLORS["School of Cardiovascular & Metabolic Health"],
        LEGACY_SCHOOL_COLORS["School of Biomedical Engineering"],
        LEGACY_SCHOOL_COLORS["School of Infection & Immunology"],
        LEGACY_SCHOOL_COLORS["School of Medicine, Dentistry & Nursing"],
        LEGACY_SCHOOL_COLORS["School of Molecular Biosciences"],
        LEGACY_SCHOOL_COLORS["School of Computing Science"],
    ]
    blue_hues = [
        extra_blue_hues[0],
        LEGACY_SCHOOL_COLORS["School of Cancer Sciences"],
        extra_blue_hues[1],
        LEGACY_SCHOOL_COLORS["School of Geographical and earth Sciences"],
        LEGACY_SCHOOL_COLORS["James Watt School of Engineering"],
        LEGACY_SCHOOL_COLORS["School of Humanities"],
        extra_blue_hues[2],
    ]

    school_color_map = dict(zip(MVLS_SCHOOLS, mvls_hues, strict=True))
    school_color_map.update(dict(zip(BLUE_HUE_SCHOOLS, blue_hues, strict=True)))

    missing = [school for school in SCHOOL_ORDER if school not in school_color_map]
    if missing:
        raise ValueError(f"Missing school colours for: {missing}")

    return school_color_map


N_UMAP_RUNS = 10


def _compute_multi_umap(embeddings, n_runs=N_UMAP_RUNS):
    """Compute n_runs UMAP projections with seeds 0..n_runs-1."""
    import umap as umap_lib
    print(f"  Computing {n_runs} UMAP projections (this may take several minutes)...")
    runs = []
    for seed in range(n_runs):
        print(f"    Run {seed + 1}/{n_runs} (seed={seed})...")
        reducer = umap_lib.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            n_components=2,
            random_state=seed,
        )
        runs.append(reducer.fit_transform(embeddings))
    return np.stack(runs, axis=0)  # [n_runs, N, 2]


def main():
    # ------------------------------------------------------------------
    # 1. Load & merge data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "glasgow_abstracts.csv"))
    authors = pd.read_csv(os.path.join(DATA_DIR, "glasgow_authors.csv"))
    coords = np.load(os.path.join(DATA_DIR, "glasgow_umap_coords.npy"))
    id_col = "paper_id" if "paper_id" in df.columns else "pmid"
    author_id_col = "paper_id" if "paper_id" in authors.columns else "pmid"

    if len(df) != len(coords):
        raise ValueError(
            "Length mismatch between abstracts and UMAP coords: "
            f"{len(df)} papers vs {len(coords)} coords. "
            "Run scripts/build_glasgow_explorer_data.py to regenerate embeddings and UMAPs."
        )

    # ------------------------------------------------------------------
    # 1b. Load or compute multi-UMAP (10 projections with different seeds)
    # ------------------------------------------------------------------
    multi_file = os.path.join(DATA_DIR, "glasgow_umap_coords_multi.npy")
    emb_file = os.path.join(DATA_DIR, "glasgow_embeddings.npy")
    multi_coords = None

    if os.path.exists(multi_file):
        _mc = np.load(multi_file)
        if _mc.shape[0] == N_UMAP_RUNS and _mc.shape[1] == len(df):
            multi_coords = _mc
            print(f"  Loaded {N_UMAP_RUNS} cached UMAP projections.")
        else:
            print(f"  Multi-UMAP cache shape {_mc.shape} doesn't match; ignoring.")

    if multi_coords is None:
        if os.path.exists(emb_file):
            print("  Loading embeddings for multi-UMAP computation...")
            embeddings = np.load(emb_file)
            multi_coords = _compute_multi_umap(embeddings)
            np.save(multi_file, multi_coords)
            print(f"  Saved multi-UMAP: {multi_coords.shape} → {multi_file}")
        else:
            print("  No embeddings found; replicating single projection for all runs.")
            multi_coords = np.stack([coords] * N_UMAP_RUNS, axis=0)

    # Attach all projection coords as df columns (survive the merge below)
    for _i in range(N_UMAP_RUNS):
        df[f"_ux{_i}"] = multi_coords[_i, :, 0].astype(np.float32)
        df[f"_uy{_i}"] = multi_coords[_i, :, 1].astype(np.float32)

    df["x"] = df["_ux0"]
    df["y"] = df["_uy0"]
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df[id_col] = df[id_col].astype(str)
    authors[author_id_col] = authors[author_id_col].astype(str)
    authors["school"] = authors["school"].map(normalize_school)
    authors = authors[authors["school"].isin(SCHOOL_ORDER)].copy()

    if authors.empty:
        raise ValueError("No Glasgow authors remain after school normalization/filtering.")

    # Citations
    if "cited_by_count" in df.columns:
        df.drop(columns=["cited_by_count"], inplace=True)
    cit_file = os.path.join(DATA_DIR, "glasgow_citations.csv")
    if os.path.exists(cit_file):
        cit = pd.read_csv(cit_file)
        cit_id_col = "paper_id" if "paper_id" in cit.columns else "pmid"
        cit[cit_id_col] = cit[cit_id_col].astype(str)
        df = df.merge(
            cit[[cit_id_col, "cited_by_count"]],
            left_on=id_col,
            right_on=cit_id_col,
            how="left",
        )
        if cit_id_col != id_col:
            df.drop(columns=[cit_id_col], inplace=True)
        df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)
    else:
        df["cited_by_count"] = 0

    # Author/school/college aggregation
    agg = authors.groupby(author_id_col).agg(
        glasgow_authors=("author_name", lambda x: "; ".join(sorted(set(x)))),
        schools=("school", lambda x: "; ".join(sorted(set(x), key=school_sort_key))),
        colleges=("college", lambda x: "; ".join(sorted(set(x)))),
        primary_college=("college", "first"),
        primary_school=("school", "first"),
    ).reset_index()
    agg[author_id_col] = agg[author_id_col].astype(str)
    df = df.merge(agg, left_on=id_col, right_on=author_id_col, how="inner")
    if author_id_col != id_col:
        df.drop(columns=[author_id_col], inplace=True)
    print(f"  Retained {len(df)} papers across {agg['primary_school'].nunique()} schools")

    # Citation graph edges
    graph_file = os.path.join(DATA_DIR, "glasgow_citation_graph.csv")
    edges_json = "[]"
    if os.path.exists(graph_file):
        edges = pd.read_csv(graph_file, dtype=str)
        if {"citing_paper_id", "cited_paper_id"}.issubset(edges.columns):
            edges = edges[["citing_paper_id", "cited_paper_id"]]
            edges_json = edges.to_json(orient="values")
            print(f"  Citation graph: {len(edges)} edges")
        elif id_col == "pmid" and {"citing_pmid", "cited_pmid"}.issubset(edges.columns):
            edges = edges[["citing_pmid", "cited_pmid"]]
            edges_json = edges.to_json(orient="values")
            print(f"  Citation graph: {len(edges)} edges")
        else:
            print("  Citation graph cache is incompatible with current paper IDs; ignoring.")

    # ------------------------------------------------------------------
    # 2. Build colour maps
    # ------------------------------------------------------------------
    school_color_map = build_school_color_map()
    picker_palette = build_picker_palette()

    # ------------------------------------------------------------------
    # 3. Prepare JSON data blob (one row per paper)
    # ------------------------------------------------------------------
    records = []
    for _, row in df.iterrows():
        records.append({
            "x": float(row["x"]),
            "y": float(row["y"]),
            "paper_id": clean_identifier(row[id_col]),
            "pmid": clean_identifier(row.get("pmid", "")),
            "openalex_id": clean_identifier(row.get("openalex_id", "")),
            "title": clean(row.get("title", "")),
            "year": clean_year(row.get("year", "")),
            "journal": clean(row.get("journal", "")),
            "doi": clean(row.get("doi", "")),
            "abstract": clean(row.get("abstract", "")),
            "all_authors": clean(row.get("all_authors", "")),
            "glasgow_authors": clean(row.get("glasgow_authors", "")),
            "cited_by_count": int(row.get("cited_by_count", 0)),
            "school": clean(row.get("primary_school", "")),
            "college": clean(row.get("primary_college", "")),
            "year_int": int(row["year_int"]) if pd.notna(row["year_int"]) else None,
        })

    data_json = json.dumps(records, separators=(",", ":"))

    # Build per-projection coordinate arrays (for the JS projection switcher)
    # Shape: [[x0,x1,...xN] for proj0, ...] same for ys
    df_reset = df.reset_index(drop=True)
    umap_xs = [
        [round(float(df_reset.at[j, f"_ux{i}"]), 3) for j in range(len(df_reset))]
        for i in range(N_UMAP_RUNS)
    ]
    umap_ys = [
        [round(float(df_reset.at[j, f"_uy{i}"]), 3) for j in range(len(df_reset))]
        for i in range(N_UMAP_RUNS)
    ]
    umap_projections_json = json.dumps({"xs": umap_xs, "ys": umap_ys}, separators=(",", ":"))

    # ------------------------------------------------------------------
    # 4. Build HTML
    # ------------------------------------------------------------------
    print("Building HTML...")
    page = _build_html(
        data_json=data_json,
        edges_json=edges_json,
        school_color_map_json=json.dumps(school_color_map, separators=(",", ":")),
        school_order_json=json.dumps(SCHOOL_ORDER, separators=(",", ":")),
        school_picker_colors_json=json.dumps(picker_palette, separators=(",", ":")),
        college_color_map_json=json.dumps(COLLEGE_COLORS, separators=(",", ":")),
        umap_projections_json=umap_projections_json,
        n_umap_runs=N_UMAP_RUNS,
        n_papers=len(df),
    )

    out_path = os.path.join(OUT_DIR, "glasgow_explorer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Saved: {out_path}  ({len(page) / 1e6:.1f} MB)")


def _build_html(*, data_json, edges_json, school_color_map_json,
                school_order_json, school_picker_colors_json,
                college_color_map_json, umap_projections_json,
                n_umap_runs, n_papers):
    projection_options = "\n    ".join(
        f'<option value="{i}">Run {i + 1}&thinsp;(seed&nbsp;{i})</option>'
        for i in range(n_umap_runs)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Glasgow Research Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#ffffff;color:#1e293b;overflow:hidden}}
.wrap{{display:grid;grid-template-columns:1fr 400px;height:100vh;transition:grid-template-columns 200ms ease}}
body.panel-hidden .wrap{{grid-template-columns:1fr 0px}}
.plot-col{{position:relative;overflow:hidden}}
#umap-plot{{width:100%;height:100%}}

/* ---- side panel ---- */
.panel{{display:flex;flex-direction:column;overflow-y:auto;background:#f8fafc;border-left:1px solid #e2e8f0;transition:transform 200ms ease,opacity 180ms ease}}
body.panel-hidden .panel{{transform:translateX(100%);opacity:0;pointer-events:none}}
.panel-head{{display:flex;align-items:center;justify-content:space-between;padding:14px 16px 10px;border-bottom:1px solid #e2e8f0}}
.panel-head h2{{margin:0;font-size:16px;font-weight:600}}
.panel-toolbar{{display:flex;gap:6px;align-items:center}}
.btn{{border:1px solid #cbd5e1;background:#ffffff;color:#1e293b;border-radius:6px;font-size:12px;padding:5px 10px;cursor:pointer}}
.btn:hover{{background:#f1f5f9}}
select.btn{{padding-right:24px}}
.control-stack{{display:flex;flex-direction:column;gap:10px;margin-bottom:14px}}
.control-block{{display:flex;flex-direction:column;gap:6px}}
.control-label{{font-size:11px;font-weight:600;letter-spacing:0.02em;text-transform:uppercase;color:#64748b}}
.slider-row{{display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid #dbe4ee;background:#ffffff;border-radius:10px}}
.slider-row input[type="range"]{{flex:1;accent-color:#0f172a}}
.slider-value{{min-width:28px;text-align:right;font-size:12px;color:#475569}}
.author-search{{display:flex;flex-direction:column;gap:8px;padding:10px 12px;border:1px solid #dbe4ee;background:#ffffff;border-radius:10px}}
.author-search-row{{display:flex;gap:6px;align-items:center}}
.author-search-input{{min-width:0;flex:1;border:1px solid #cbd5e1;border-radius:6px;padding:6px 8px;font-size:12px;color:#0f172a;background:#ffffff}}
.author-search-input:focus{{outline:2px solid rgba(15,23,42,0.16);outline-offset:1px}}
.author-results{{display:flex;flex-direction:column;gap:5px;max-height:180px;overflow-y:auto}}
.author-result{{display:grid;grid-template-columns:8px minmax(0,1fr) auto;align-items:center;gap:8px;border:1px solid #e2e8f0;background:#f8fafc;border-radius:7px;padding:6px 7px;cursor:pointer;text-align:left;color:#1e293b}}
.author-result:hover{{background:#f1f5f9;border-color:#cbd5e1}}
.author-result.active{{border-color:#0f172a;background:#eef2f7}}
.author-result-swatch{{width:8px;height:28px;border-radius:999px}}
.author-result-name{{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.author-result-meta{{font-size:11px;color:#64748b;white-space:nowrap}}
.author-empty{{font-size:12px;color:#94a3b8;padding:2px 1px}}
.toggle-row{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border:1px solid #dbe4ee;background:#ffffff;border-radius:10px}}
.toggle-copy{{display:flex;flex-direction:column;gap:2px}}
.toggle-title{{font-size:13px;color:#1e293b}}
.toggle-sub{{font-size:11px;color:#64748b}}
.switch{{position:relative;width:40px;height:22px;flex-shrink:0}}
.switch input{{position:absolute;inset:0;opacity:0;cursor:pointer}}
.switch-track{{position:absolute;inset:0;border-radius:999px;background:#cbd5e1;transition:background 120ms ease}}
.switch-thumb{{position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:999px;background:#ffffff;box-shadow:0 1px 2px rgba(15,23,42,0.2);transition:transform 120ms ease}}
.switch input:checked + .switch-track{{background:#0f172a}}
.switch input:checked + .switch-track .switch-thumb{{transform:translateX(18px)}}
.panel-pop{{position:fixed;right:10px;top:50%;transform:translateY(-50%);z-index:20;border:1px solid #cbd5e1;background:#ffffff;color:#1e293b;border-radius:999px;padding:8px 14px;font-size:12px;cursor:pointer;display:none}}
body.panel-hidden .panel-pop{{display:block}}

.panel-body{{flex:1;padding:16px;overflow-y:auto}}
.instructions{{font-size:13px;color:#64748b;margin-bottom:14px}}
.paper-card{{border-radius:10px;padding:14px;margin-bottom:10px;background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #cbd5e1}}
.paper-card h3{{margin:0 0 8px;font-size:15px;line-height:1.4;color:#0f172a}}
.meta-row{{font-size:12px;color:#475569;margin-bottom:4px}}
.meta-row strong{{color:#1e293b}}
.abstract-text{{font-size:13px;line-height:1.55;color:#334155;white-space:pre-wrap;margin-top:10px}}
a{{color:#2563eb;text-decoration:none}}
a:hover{{text-decoration:underline}}
.count-badge{{display:inline-block;background:#f1f5f9;color:#64748b;border-radius:9999px;padding:2px 8px;font-size:11px;margin-left:6px}}

/* legend at bottom-left of plot */
#colour-legend{{position:absolute;bottom:12px;left:12px;background:rgba(255,255,255,0.92);border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;max-height:50vh;overflow-y:auto;font-size:11px;z-index:10;max-width:260px}}
#colour-legend .leg-title{{font-weight:600;margin-bottom:6px;font-size:12px;color:#1e293b}}
.leg-group-title{{font-weight:700;margin:9px 0 5px;font-size:11px;letter-spacing:0.04em;text-transform:uppercase;color:#0f172a}}
.leg-group-title:first-of-type{{margin-top:0}}
.leg-bulk{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:0 0 8px}}
.leg-bulk-row{{display:flex;align-items:center;gap:5px;border:1px solid #e2e8f0;background:#f8fafc;border-radius:6px;padding:4px 6px;color:#334155;cursor:pointer;min-width:0}}
.leg-bulk-row:hover{{background:#f1f5f9}}
.leg-bulk-row input{{margin:0;accent-color:#0f172a;flex-shrink:0}}
.leg-bulk-label{{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.leg-item{{display:flex;align-items:center;gap:6px;margin-bottom:3px;opacity:0.9}}
.leg-item:hover{{opacity:1}}
.leg-swatch-btn{{width:14px;height:14px;border-radius:3px;flex-shrink:0;border:1px solid rgba(15,23,42,0.18);padding:0;display:inline-flex;align-items:center;justify-content:center;cursor:pointer;color:#ffffff;font-size:10px;line-height:1;background-clip:padding-box}}
.leg-swatch-btn.off{{opacity:0.35}}
.leg-swatch-check{{font-weight:700;transform:translateY(-0.5px)}}
.leg-label{{color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.leg-label-btn{{border:none;background:none;padding:0;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font:inherit;text-align:left;cursor:pointer}}
.leg-label-btn:hover{{color:#0f172a;text-decoration:underline}}

/* palette modal */
.palette-backdrop{{position:fixed;inset:0;background:rgba(15,23,42,0.34);display:none;align-items:center;justify-content:center;z-index:40;padding:18px}}
.palette-backdrop.open{{display:flex}}
.palette-dialog{{width:min(920px,100%);max-height:min(82vh,760px);overflow:hidden;background:#ffffff;border-radius:16px;border:1px solid #cbd5e1;box-shadow:0 25px 60px rgba(15,23,42,0.22);display:flex;flex-direction:column}}
.palette-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:16px 18px 12px;border-bottom:1px solid #e2e8f0}}
.palette-head h3{{margin:0;font-size:16px;color:#0f172a}}
.palette-sub{{font-size:12px;color:#64748b;margin-top:4px}}
.palette-actions{{display:flex;gap:8px;align-items:center}}
.palette-body{{padding:14px 18px 18px;overflow:auto}}
.palette-current{{display:flex;align-items:center;gap:10px;font-size:13px;color:#334155;margin-bottom:12px}}
.palette-current-swatch{{width:18px;height:18px;border-radius:4px;border:1px solid rgba(15,23,42,0.18);flex-shrink:0}}
.palette-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(48px,1fr));gap:8px}}
.palette-chip{{border:1px solid rgba(15,23,42,0.12);border-radius:10px;height:44px;cursor:pointer;position:relative;transition:transform 120ms ease,border-color 120ms ease,box-shadow 120ms ease}}
.palette-chip:hover{{transform:translateY(-1px);border-color:#94a3b8}}
.palette-chip.active{{border-color:#0f172a;box-shadow:0 0 0 2px rgba(15,23,42,0.15)}}
.palette-chip-label{{position:absolute;right:5px;bottom:4px;font-size:9px;color:rgba(255,255,255,0.92);text-shadow:0 1px 2px rgba(15,23,42,0.7)}}
</style>
</head>
<body>
<button id="panel-pop" class="panel-pop">&#9664; Details</button>
<div class="wrap">
  <div class="plot-col">
    <div id="umap-plot"></div>
    <div id="colour-legend"></div>
  </div>
  <aside class="panel">
    <div class="panel-head">
      <h2>Paper Details</h2>
      <div class="panel-toolbar">
        <button id="panel-toggle" class="btn" type="button">Hide &#9654;</button>
      </div>
    </div>
    <div class="panel-body">
      <div class="control-stack">
        <div class="control-block">
          <label class="control-label" for="colour-mode">Colour</label>
          <select id="colour-mode" class="btn">
            <option value="school">School</option>
            <option value="college">College</option>
            <option value="year">Year</option>
          </select>
        </div>
        <div class="control-block">
          <label class="control-label" for="projection-select">UMAP Projection</label>
          <select id="projection-select" class="btn">
            {projection_options}
          </select>
        </div>
        <div class="control-block">
          <label class="control-label" for="interaction-mode">Interaction</label>
          <select id="interaction-mode" class="btn">
            <option value="pan">Move map</option>
            <option value="zoom">Zoom box</option>
          </select>
        </div>
        <div class="control-block">
          <div class="control-label">Citation Network</div>
          <label class="toggle-row" for="citation-network-toggle">
            <span class="toggle-copy">
              <span class="toggle-title">Show citation links</span>
              <span class="toggle-sub">Overlay the full citation network on top of the current colour.</span>
            </span>
            <span class="switch">
              <input id="citation-network-toggle" type="checkbox" />
              <span class="switch-track"><span class="switch-thumb"></span></span>
            </span>
          </label>
        </div>
        <div class="control-block">
          <div class="control-label">Filter</div>
          <label class="toggle-row" for="imaging-only-toggle">
            <span class="toggle-copy">
              <span class="toggle-title">Imaging work only</span>
              <span class="toggle-sub">Show only papers using imaging, electrophysiology, or brain stimulation techniques.</span>
            </span>
            <span class="switch">
              <input id="imaging-only-toggle" type="checkbox" />
              <span class="switch-track"><span class="switch-thumb"></span></span>
            </span>
          </label>
        </div>
        <div class="control-block">
          <div class="control-label">Point Size</div>
          <label class="slider-row" for="point-size-slider">
            <input id="point-size-slider" type="range" min="1" max="12" step="1" value="4" />
            <span id="point-size-value" class="slider-value">4</span>
          </label>
        </div>
        <div class="control-block">
          <label class="control-label" for="author-search-input">Author</label>
          <div class="author-search">
            <div class="author-search-row">
              <input id="author-search-input" class="author-search-input" type="search" placeholder="Search author" autocomplete="off" />
              <button id="author-search-reset" class="btn" type="button">Reset</button>
            </div>
            <div id="author-search-results" class="author-results"></div>
          </div>
        </div>
      </div>
      <div class="instructions">Hover for preview &middot; Click to pin details &middot; Use the menu to change colour or projection</div>
      <div id="paper-detail" style="color:#94a3b8;">Click a point to see its details.</div>
    </div>
  </aside>
</div>
<div id="palette-backdrop" class="palette-backdrop" aria-hidden="true">
  <div class="palette-dialog" role="dialog" aria-modal="true" aria-labelledby="palette-title">
    <div class="palette-head">
      <div>
        <h3 id="palette-title">Choose School Colour</h3>
        <div id="palette-subtitle" class="palette-sub">Select one of the first 100 distinguishable colours, ordered by hue family and brightness.</div>
      </div>
      <div class="palette-actions">
        <button id="palette-reset" class="btn" type="button">Reset School</button>
        <button id="palette-reset-all" class="btn" type="button">Reset All</button>
        <button id="palette-close" class="btn" type="button">Close</button>
      </div>
    </div>
    <div class="palette-body">
      <div class="palette-current">
        <span id="palette-current-swatch" class="palette-current-swatch"></span>
        <span id="palette-current-label"></span>
      </div>
      <div id="palette-grid" class="palette-grid"></div>
    </div>
  </div>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────
const DATA = {data_json};
const EDGES = {edges_json};
const DEFAULT_SCHOOL_COLORS = {school_color_map_json};
const SCHOOL_COLORS = {{ ...DEFAULT_SCHOOL_COLORS }};
const SCHOOL_ORDER = {school_order_json};
const SCHOOL_PICKER_COLORS = {school_picker_colors_json};
const COLLEGE_COLORS = {college_color_map_json};
const UMAP_PROJECTIONS = {umap_projections_json};  // {{xs: [[...], ...], ys: [[...], ...]}}
let currentProjection = 0;
const N = DATA.length;
const presentSchools = new Set(DATA.map(d => d.school).filter(Boolean));
const presentColleges = new Set(DATA.map(d => d.college).filter(Boolean));
const presentYears = Array.from(new Set(DATA.map(d => d.year_int).filter(y => y !== null))).sort((a, b) => a - b);
const MVLS_SCHOOL_SET = new Set(SCHOOL_ORDER.slice(0, SCHOOL_ORDER.indexOf('School of Mathematics and Statistics')));
const OTHER_SCHOOL_SET = new Set(SCHOOL_ORDER.filter(school => !MVLS_SCHOOL_SET.has(school)));
const SCHOOL_COLOR_STORAGE_KEY = 'glasgow-explorer-school-colors-v2';
const SCHOOL_VISIBILITY_STORAGE_KEY = 'glasgow-explorer-school-visibility-v1';
const COLLEGE_VISIBILITY_STORAGE_KEY = 'glasgow-explorer-college-visibility-v1';
const YEAR_VISIBILITY_STORAGE_KEY = 'glasgow-explorer-year-visibility-v1';
const CITATION_NETWORK_STORAGE_KEY = 'glasgow-explorer-citation-network-v1';
const IMAGING_ONLY_STORAGE_KEY = 'glasgow-explorer-imaging-only-v1';
const POINT_SIZE_STORAGE_KEY = 'glasgow-explorer-point-size-v1';
const INTERACTION_MODE_STORAGE_KEY = 'glasgow-explorer-interaction-mode-v1';

const IMAGING_KEYWORDS = [
  'TMS','transcranial magnetic stimulation',
  'tACS','transcranial alternating current stimulation',
  'tDCS','transcranial direct current stimulation',
  'tRNS','transcranial random noise stimulation',
  'transcranial stimulation',
  'fMRI','functional MRI','functional magnetic resonance imaging',
  'magnetic resonance imaging','MRI','BOLD',
  'DTI','diffusion tensor imaging','DWI','diffusion-weighted',
  'tractography','resting-state','rs-fMRI',
  'MRS','magnetic resonance spectroscopy','voxel-based morphometry','VBM',
  'EEG','electroencephalograph','MEG','magnetoencephalograph',
  'event-related potential','ERP',
  'ECoG','electrocorticograph','iEEG','intracranial EEG',
  'sEEG','stereoelectroencephalograph','LFP','local field potential',
  'single-unit','multi-unit','multiunit','electrophysiology',
  'spike sorting','patch clamp','tetrode','silicon probe',
  'Neuropixels','microelectrode','neural recording',
  'PET','positron emission tomography','SPECT','single-photon emission',
  'radiotracer','radiolabeled','autoradiography',
  'fNIRS','functional near-infrared','NIRS','near-infrared spectroscopy',
  'optical imaging','intrinsic signal','intrinsic optical',
  'calcium imaging','voltage imaging','widefield imaging',
  'microscopy','two-photon','2-photon','multiphoton','confocal',
  'fluorescence microscopy','light-sheet','STED','cryostat',
  'immunohistochemistry','immunofluorescence','histology','Nissl',
  'ultrasound','functional ultrasound','fUS','ultrasonography',
  'CT scan','computed tomography','X-ray','radiograph','angiography',
  'stereotaxic','stereotactic','cytoarchitect',
];
const IMAGING_REGEX = new RegExp(
  '\\\\b(?:' + IMAGING_KEYWORDS.map(k => k.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&')).join('|') + ')\\\\b',
  'i'
);
let selectedPointIndex = null;
let activePaletteSchool = null;
const hiddenSchools = new Set();
const hiddenColleges = new Set();
const hiddenYears = new Set();
let citationNetworkEnabled = false;
let imagingOnlyEnabled = false;
let pointSize = 4;
let interactionMode = 'pan';
let activeAuthorName = '';

const imagingMask = DATA.map(d => IMAGING_REGEX.test(d.abstract || ''));

function splitAuthors(value) {{
  return String(value || '')
    .split(';')
    .map(name => name.trim())
    .filter(Boolean);
}}

function normalizeSearch(value) {{
  return String(value || '')
    .normalize('NFKD')
    .replace(/[\\u0300-\\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}}

const datumGlasgowAuthors = DATA.map(d => splitAuthors(d.glasgow_authors));
const authorSummaries = new Map();
DATA.forEach((d, idx) => {{
  datumGlasgowAuthors[idx].forEach(author => {{
    if (!authorSummaries.has(author)) {{
      authorSummaries.set(author, {{
        name: author,
        norm: normalizeSearch(author),
        count: 0,
        schools: new Map(),
      }});
    }}
    const summary = authorSummaries.get(author);
    summary.count += 1;
    summary.schools.set(d.school, (summary.schools.get(d.school) || 0) + 1);
  }});
}});

const authorList = Array.from(authorSummaries.values()).map(summary => {{
  const schools = Array.from(summary.schools.entries()).sort((a, b) => b[1] - a[1]);
  const primarySchool = schools[0]?.[0] || '';
  return {{ ...summary, primarySchool, schoolCount: schools.length }};
}}).sort((a, b) => a.name.localeCompare(b.name));
const authorByName = new Map(authorList.map(summary => [summary.name, summary]));

// pre-index
const paperIdx = {{}};
DATA.forEach((d, i) => {{ paperIdx[d.paper_id] = i; }});

// citation adjacency
const citesOut = {{}};   // paper_id -> [paper_id, ...]
const citedBy = {{}};    // paper_id -> [paper_id, ...]
EDGES.forEach(e => {{
  const [a, b] = e;
  if (!citesOut[a]) citesOut[a] = [];
  citesOut[a].push(b);
  if (!citedBy[b]) citedBy[b] = [];
  citedBy[b].push(a);
}});

// year colour scale
const years = DATA.map(d => d.year_int).filter(y => y !== null);
const minYear = Math.min(...years);
const maxYear = Math.max(...years);
function yearColor(y) {{
  if (y === null) return 'rgb(80,80,80)';
  const t = (y - minYear) / Math.max(maxYear - minYear, 1);
  // viridis-ish: purple → teal → yellow
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + (0.5 - Math.abs(t - 0.5)) * 2 * (150 - 84) + t * (37 - 84));
  return `rgb(${{r}},${{g}},${{Math.max(0, Math.min(255, b))}})`;
}}

// citation connection count
const connCount = DATA.map(d => {{
  return (citesOut[d.paper_id] || []).length + (citedBy[d.paper_id] || []).length;
}});
const maxConn = Math.max(1, ...connCount);
function connColor(n) {{
  const t = Math.sqrt(n / maxConn);
  const r = Math.round(255 * t);
  const g = Math.round(255 * (1 - t) * 0.4);
  return `rgb(${{r}},${{g}},40)`;
}}

function parseCssColor(color) {{
  if (!color) return null;
  const rgbMatch = color.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
  if (rgbMatch) {{
    return rgbMatch.slice(1, 4).map(value => Math.max(0, Math.min(255, Number(value))));
  }}
  const hexMatch = color.match(/^#([0-9a-f]{{3}}|[0-9a-f]{{6}})$/i);
  if (hexMatch) {{
    let hex = hexMatch[1];
    if (hex.length === 3) hex = hex.split('').map(ch => ch + ch).join('');
    return [0, 2, 4].map(pos => parseInt(hex.slice(pos, pos + 2), 16));
  }}
  return null;
}}

function paleAuthorContextColor(color) {{
  const rgb = parseCssColor(color);
  if (!rgb) return color;
  const grey = [203, 213, 225];
  const mixed = rgb.map((channel, idx) => Math.round(channel * 0.5 + grey[idx] * 0.5));
  return `rgb(${{mixed[0]}},${{mixed[1]}},${{mixed[2]}})`;
}}

function loadStoredSchoolColors() {{
  try {{
    const raw = localStorage.getItem(SCHOOL_COLOR_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    Object.entries(parsed).forEach(([school, color]) => {{
      if (Object.prototype.hasOwnProperty.call(DEFAULT_SCHOOL_COLORS, school) && typeof color === 'string') {{
        SCHOOL_COLORS[school] = color;
      }}
    }});
  }} catch (_err) {{
    // Ignore malformed browser storage.
  }}
}}

function persistSchoolColors() {{
  try {{
    localStorage.setItem(SCHOOL_COLOR_STORAGE_KEY, JSON.stringify(SCHOOL_COLORS));
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadHiddenSchools() {{
  try {{
    const raw = localStorage.getItem(SCHOOL_VISIBILITY_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    parsed.forEach(school => {{
      if (presentSchools.has(school)) hiddenSchools.add(school);
    }});
  }} catch (_err) {{
    // Ignore malformed browser storage.
  }}
}}

function persistHiddenSchools() {{
  try {{
    localStorage.setItem(SCHOOL_VISIBILITY_STORAGE_KEY, JSON.stringify(Array.from(hiddenSchools)));
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadHiddenColleges() {{
  try {{
    const raw = localStorage.getItem(COLLEGE_VISIBILITY_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    parsed.forEach(college => {{
      if (presentColleges.has(college)) hiddenColleges.add(college);
    }});
  }} catch (_err) {{
    // Ignore malformed browser storage.
  }}
}}

function persistHiddenColleges() {{
  try {{
    localStorage.setItem(COLLEGE_VISIBILITY_STORAGE_KEY, JSON.stringify(Array.from(hiddenColleges)));
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadHiddenYears() {{
  try {{
    const raw = localStorage.getItem(YEAR_VISIBILITY_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    parsed.forEach(year => {{
      const normalized = String(year);
      if (presentYears.includes(Number(normalized))) hiddenYears.add(normalized);
    }});
  }} catch (_err) {{
    // Ignore malformed browser storage.
  }}
}}

function persistHiddenYears() {{
  try {{
    localStorage.setItem(YEAR_VISIBILITY_STORAGE_KEY, JSON.stringify(Array.from(hiddenYears)));
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadCitationNetworkEnabled() {{
  try {{
    citationNetworkEnabled = localStorage.getItem(CITATION_NETWORK_STORAGE_KEY) === 'true';
  }} catch (_err) {{
    citationNetworkEnabled = false;
  }}
}}

function persistCitationNetworkEnabled() {{
  try {{
    localStorage.setItem(CITATION_NETWORK_STORAGE_KEY, citationNetworkEnabled ? 'true' : 'false');
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadImagingOnlyEnabled() {{
  try {{
    imagingOnlyEnabled = localStorage.getItem(IMAGING_ONLY_STORAGE_KEY) === 'true';
  }} catch (_err) {{
    imagingOnlyEnabled = false;
  }}
}}

function persistImagingOnlyEnabled() {{
  try {{
    localStorage.setItem(IMAGING_ONLY_STORAGE_KEY, imagingOnlyEnabled ? 'true' : 'false');
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadPointSize() {{
  try {{
    const raw = localStorage.getItem(POINT_SIZE_STORAGE_KEY);
    if (!raw) return;
    const parsed = Number(raw);
    if (!Number.isNaN(parsed)) pointSize = Math.max(1, Math.min(12, parsed));
  }} catch (_err) {{
    pointSize = 4;
  }}
}}

function persistPointSize() {{
  try {{
    localStorage.setItem(POINT_SIZE_STORAGE_KEY, String(pointSize));
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

function loadInteractionMode() {{
  try {{
    const raw = localStorage.getItem(INTERACTION_MODE_STORAGE_KEY);
    if (raw === 'pan' || raw === 'zoom') interactionMode = raw;
  }} catch (_err) {{
    interactionMode = 'pan';
  }}
}}

function persistInteractionMode() {{
  try {{
    localStorage.setItem(INTERACTION_MODE_STORAGE_KEY, interactionMode);
  }} catch (_err) {{
    // Ignore browsers that block storage.
  }}
}}

loadStoredSchoolColors();
loadHiddenSchools();
loadHiddenColleges();
loadHiddenYears();
loadCitationNetworkEnabled();
loadImagingOnlyEnabled();
loadPointSize();
loadInteractionMode();

function isSchoolVisible(school) {{
  return !hiddenSchools.has(school);
}}

function schoolGroupMembers(group) {{
  const groupSet = group === 'mvls' ? MVLS_SCHOOL_SET : OTHER_SCHOOL_SET;
  return SCHOOL_ORDER.filter(school => presentSchools.has(school) && groupSet.has(school));
}}

function schoolGroupVisibility(group) {{
  const members = schoolGroupMembers(group);
  const visible = members.filter(isSchoolVisible).length;
  return {{ members, visible }};
}}

function setSchoolGroupVisible(group, visible) {{
  schoolGroupMembers(group).forEach(school => {{
    if (visible) {{
      hiddenSchools.delete(school);
    }} else {{
      hiddenSchools.add(school);
    }}
  }});
  persistHiddenSchools();
  applyColourState();
}}

function isCollegeVisible(college) {{
  return !hiddenColleges.has(college);
}}

function yearKey(value) {{
  return value === null ? '' : String(value);
}}

function isYearVisible(year) {{
  const key = yearKey(year);
  return key ? !hiddenYears.has(key) : true;
}}

function isDatumVisible(d, mode, idx = null) {{
  if (!isSchoolVisible(d.school)) return false;
  const index = idx ?? paperIdx[d.paper_id];
  if (imagingOnlyEnabled && (index === undefined || !imagingMask[index])) return false;
  if (mode === 'college') return isCollegeVisible(d.college);
  if (mode === 'year') return isYearVisible(d.year_int);
  return true;
}}

function getCurrentMode() {{
  const select = document.getElementById('colour-mode');
  return select ? select.value : 'school';
}}

// ── colour assignment helpers ────────────────────────────────────
function datumColor(d, mode, idx) {{
  if (!isDatumVisible(d, mode, idx)) return 'rgba(0,0,0,0)';
  let color = 'rgb(100,100,100)';
  if (mode === 'school') color = SCHOOL_COLORS[d.school] || 'rgb(80,80,80)';
  if (mode === 'college') color = COLLEGE_COLORS[d.college] || 'rgb(80,80,80)';
  if (mode === 'year') color = yearColor(d.year_int);
  if (mode === 'citations') color = SCHOOL_COLORS[d.school] || 'rgb(180,180,180)';
  const isAuthorContext = activeAuthorName && !datumGlasgowAuthors[idx].includes(activeAuthorName);
  return isAuthorContext ? paleAuthorContextColor(color) : color;
}}

function getColors(mode) {{
  return DATA.map((d, idx) => datumColor(d, mode, idx));
}}

function getMarkerSizes() {{
  const mode = getCurrentMode();
  return DATA.map((d, idx) => {{
    if (!isDatumVisible(d, mode, idx)) return 0.01;
    const isAuthorHit = activeAuthorName && datumGlasgowAuthors[idx].includes(activeAuthorName);
    return isAuthorHit ? pointSize + 3 : pointSize;
  }});
}}

function getLegendItems(mode) {{
  if (mode === 'school') {{
    return SCHOOL_ORDER
      .filter(name => presentSchools.has(name))
      .map(name => [name, SCHOOL_COLORS[name], name]);
  }}
  if (mode === 'college') {{
    return Object.entries(COLLEGE_COLORS)
      .filter(([name]) => presentColleges.has(name))
      .sort((a,b) => a[0].localeCompare(b[0]));
  }}
  if (mode === 'year') {{
    return presentYears.map(year => [String(year), yearColor(year), String(year)]);
  }}
  return [];
}}

// ── build initial plot ───────────────────────────────────────────
const xs = DATA.map(d => d.x);
const ys = DATA.map(d => d.y);
const texts = DATA.map(d => (d.title || '').slice(0, 80) + '...');

const scatterTrace = {{
  x: xs, y: ys,
  mode: 'markers',
  type: 'scattergl',
  marker: {{ size: getMarkerSizes(), opacity: 0.5, color: getColors('school') }},
  text: texts,
  hovertemplate: '<b>%{{text}}</b><extra></extra>',
  hoverinfo: 'text',
}};

const layout = {{
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  margin: {{ l: 5, r: 5, t: 5, b: 5 }},
  xaxis: {{ visible: false }},
  yaxis: {{ visible: false }},
  showlegend: false,
  hovermode: 'closest',
  dragmode: interactionMode,
}};

Plotly.newPlot('umap-plot', [scatterTrace], layout, {{
  responsive: true,
  displayModeBar: false,
  scrollZoom: true,
}});

// ── edge traces management ───────────────────────────────────────
let edgeTraceCount = 0;
function clearEdges() {{
  if (edgeTraceCount > 0) {{
    const indices = [];
    const total = document.getElementById('umap-plot').data.length;
    for (let i = total - edgeTraceCount; i < total; i++) indices.push(i);
    Plotly.deleteTraces('umap-plot', indices);
    edgeTraceCount = 0;
  }}
}}

function isEdgeVisible(sourcePaperId, targetPaperId) {{
  const mode = getCurrentMode();
  const sourceIdx = paperIdx[sourcePaperId];
  const targetIdx = paperIdx[targetPaperId];
  if (sourceIdx === undefined || targetIdx === undefined) return false;
  return isDatumVisible(DATA[sourceIdx], mode) && isDatumVisible(DATA[targetIdx], mode);
}}

function buildSelectedEdgeTraces(paperId) {{
  const srcIdx = paperIdx[paperId];
  if (srcIdx === undefined || !isDatumVisible(DATA[srcIdx], getCurrentMode(), srcIdx)) return [];

  const sx = DATA[srcIdx].x;
  const sy = DATA[srcIdx].y;
  const traces = [];

  const outs = citesOut[paperId] || [];
  if (outs.length) {{
    const ex = [], ey = [];
    outs.forEach(t => {{
      const ti = paperIdx[t];
      if (ti !== undefined && isEdgeVisible(paperId, t)) {{
        ex.push(sx, DATA[ti].x, null);
        ey.push(sy, DATA[ti].y, null);
      }}
    }});
    if (ex.length) traces.push({{
      x: ex, y: ey, mode: 'lines', type: 'scatter',
      line: {{ color: 'rgba(59,130,246,0.72)', width: 1.5 }},
      hoverinfo: 'skip', showlegend: false,
    }});
  }}

  const ins = citedBy[paperId] || [];
  if (ins.length) {{
    const ex = [], ey = [];
    ins.forEach(s => {{
      const si = paperIdx[s];
      if (si !== undefined && isEdgeVisible(s, paperId)) {{
        ex.push(DATA[si].x, sx, null);
        ey.push(DATA[si].y, sy, null);
      }}
    }});
    if (ex.length) traces.push({{
      x: ex, y: ey, mode: 'lines', type: 'scatter',
      line: {{ color: 'rgba(239,68,68,0.72)', width: 1.5 }},
      hoverinfo: 'skip', showlegend: false,
    }});
  }}

  return traces;
}}

function drawSelectedEdges(paperId) {{
  clearEdges();
  const traces = buildSelectedEdgeTraces(paperId);
  if (traces.length) {{
    Plotly.addTraces('umap-plot', traces);
    edgeTraceCount = traces.length;
  }}
}}

function drawCitationNetwork(selectedPaperId = null) {{
  clearEdges();
  const ex = [];
  const ey = [];
  EDGES.forEach(e => {{
    const [source, target] = e;
    if (!isEdgeVisible(source, target)) return;
    const sourceIdx = paperIdx[source];
    const targetIdx = paperIdx[target];
    ex.push(DATA[sourceIdx].x, DATA[targetIdx].x, null);
    ey.push(DATA[sourceIdx].y, DATA[targetIdx].y, null);
  }});

  const traces = [];
  if (ex.length) {{
    traces.push({{
      x: ex, y: ey, mode: 'lines', type: 'scatter',
      line: {{ color: 'rgba(71,85,105,0.14)', width: 1.0 }},
      hoverinfo: 'skip', showlegend: false,
    }});
  }}

  if (selectedPaperId) {{
    traces.push(...buildSelectedEdgeTraces(selectedPaperId));
  }}

  if (traces.length) {{
    Plotly.addTraces('umap-plot', traces);
    edgeTraceCount = traces.length;
  }}
}}

// ── legend ────────────────────────────────────────────────────────
const legendEl = document.getElementById('colour-legend');
function renderLegend(mode) {{
  const items = getLegendItems(mode);
  const modeLabel = {{ school: 'School', college: 'College', year: 'Year' }}[mode] || mode;
  legendEl.innerHTML = '';
  const title = document.createElement('div');
  title.className = 'leg-title';
  title.textContent = mode === 'school' ? `${{modeLabel}} (click a name to recolour)` : `${{modeLabel}} (tick to show or hide)`;
  legendEl.appendChild(title);

  if (mode === 'school') {{
    const bulk = document.createElement('div');
    bulk.className = 'leg-bulk';
    [
      ['mvls', 'MVLS'],
      ['other', 'Other'],
    ].forEach(([group, label]) => {{
      const state = schoolGroupVisibility(group);
      const bulkLabel = document.createElement('label');
      bulkLabel.className = 'leg-bulk-row';
      bulkLabel.title = `Toggle all ${{label}} schools`;
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.dataset.schoolGroupToggle = group;
      checkbox.checked = state.members.length > 0 && state.visible === state.members.length;
      checkbox.indeterminate = state.visible > 0 && state.visible < state.members.length;
      checkbox.disabled = state.members.length === 0;
      const text = document.createElement('span');
      text.className = 'leg-bulk-label';
      text.textContent = label;
      bulkLabel.appendChild(checkbox);
      bulkLabel.appendChild(text);
      bulk.appendChild(bulkLabel);
    }});
    legendEl.appendChild(bulk);

    const mvlsHeader = document.createElement('div');
    mvlsHeader.className = 'leg-group-title';
    mvlsHeader.textContent = 'MVLS';
    legendEl.appendChild(mvlsHeader);
  }}

  items.forEach(([label, color, rawValue]) => {{
    if (mode === 'school' && label === 'School of Mathematics and Statistics') {{
      const otherHeader = document.createElement('div');
      otherHeader.className = 'leg-group-title';
      otherHeader.textContent = 'Other';
      legendEl.appendChild(otherHeader);
    }}

    const item = document.createElement('div');
    item.className = 'leg-item';

    const swatch = document.createElement((mode === 'school' || mode === 'college' || mode === 'year') ? 'button' : 'span');
    if (mode === 'school' || mode === 'college' || mode === 'year') {{
      swatch.type = 'button';
    }}
    swatch.className = 'leg-swatch-btn';
    if (mode === 'school' && !isSchoolVisible(label)) swatch.classList.add('off');
    if (mode === 'college' && !isCollegeVisible(label)) swatch.classList.add('off');
    if (mode === 'year' && !isYearVisible(rawValue)) swatch.classList.add('off');
    swatch.style.background = color;
    if (mode === 'school') {{
      swatch.dataset.schoolToggle = label;
      swatch.title = isSchoolVisible(label) ? `Hide ${{label}}` : `Show ${{label}}`;
      const check = document.createElement('span');
      check.className = 'leg-swatch-check';
      check.textContent = isSchoolVisible(label) ? '✓' : '';
      swatch.appendChild(check);
    }} else if (mode === 'college') {{
      swatch.dataset.collegeToggle = label;
      swatch.title = isCollegeVisible(label) ? `Hide ${{label}}` : `Show ${{label}}`;
      const check = document.createElement('span');
      check.className = 'leg-swatch-check';
      check.textContent = isCollegeVisible(label) ? '✓' : '';
      swatch.appendChild(check);
    }} else if (mode === 'year') {{
      swatch.dataset.yearToggle = rawValue;
      swatch.title = isYearVisible(rawValue) ? `Hide ${{label}}` : `Show ${{label}}`;
      const check = document.createElement('span');
      check.className = 'leg-swatch-check';
      check.textContent = isYearVisible(rawValue) ? '✓' : '';
      swatch.appendChild(check);
    }}
    item.appendChild(swatch);

    if (mode === 'school') {{
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'leg-label-btn';
      button.dataset.school = label;
      button.textContent = label;
      item.appendChild(button);
    }} else {{
      const labelEl = document.createElement('span');
      labelEl.className = 'leg-label';
      labelEl.textContent = label;
      item.appendChild(labelEl);
    }}

    legendEl.appendChild(item);
  }});
}}
renderLegend('school');

const paletteBackdrop = document.getElementById('palette-backdrop');
const paletteGrid = document.getElementById('palette-grid');
const paletteSubtitle = document.getElementById('palette-subtitle');
const paletteCurrentSwatch = document.getElementById('palette-current-swatch');
const paletteCurrentLabel = document.getElementById('palette-current-label');
const paletteClose = document.getElementById('palette-close');
const paletteReset = document.getElementById('palette-reset');
const paletteResetAll = document.getElementById('palette-reset-all');
const citationNetworkToggle = document.getElementById('citation-network-toggle');
const pointSizeSlider = document.getElementById('point-size-slider');
const pointSizeValue = document.getElementById('point-size-value');
const interactionModeSelect = document.getElementById('interaction-mode');
const imagingOnlyToggle = document.getElementById('imaging-only-toggle');
const authorSearchInput = document.getElementById('author-search-input');
const authorSearchResults = document.getElementById('author-search-results');
const authorSearchReset = document.getElementById('author-search-reset');
citationNetworkToggle.checked = citationNetworkEnabled;
imagingOnlyToggle.checked = imagingOnlyEnabled;
pointSizeSlider.value = String(pointSize);
pointSizeValue.textContent = String(pointSize);
interactionModeSelect.value = interactionMode;

function authorResultScore(query, summary) {{
  if (!query) return 0;
  const name = summary.norm;
  const words = name.split(' ');
  if (name === query) return 1000 + summary.count;
  if (name.startsWith(query)) return 850 + summary.count;
  if (words.some(word => word.startsWith(query))) return 700 + summary.count;
  if (name.includes(query)) return 550 + summary.count;
  const parts = query.split(' ').filter(Boolean);
  if (parts.length > 1 && parts.every(part => name.includes(part))) return 450 + summary.count;
  let pos = 0;
  for (const ch of query.replace(/\\s+/g, '')) {{
    pos = name.indexOf(ch, pos);
    if (pos === -1) return 0;
    pos += 1;
  }}
  return query.length >= 3 ? 250 + summary.count : 0;
}}

function renderAuthorResults() {{
  const query = normalizeSearch(authorSearchInput.value);
  authorSearchResults.innerHTML = '';
  if (!query) {{
    if (activeAuthorName) {{
      const active = authorByName.get(activeAuthorName);
      if (active) renderAuthorResultButton(active);
    }} else {{
      const empty = document.createElement('div');
      empty.className = 'author-empty';
      empty.textContent = 'No author selected.';
      authorSearchResults.appendChild(empty);
    }}
    return;
  }}

  const matches = authorList
    .map(summary => [summary, authorResultScore(query, summary)])
    .filter(([, score]) => score > 0)
    .sort((a, b) => b[1] - a[1] || a[0].name.localeCompare(b[0].name))
    .slice(0, 8);

  if (!matches.length) {{
    const empty = document.createElement('div');
    empty.className = 'author-empty';
    empty.textContent = 'No matches.';
    authorSearchResults.appendChild(empty);
    return;
  }}

  matches.forEach(([summary]) => renderAuthorResultButton(summary));
}}

function renderAuthorResultButton(summary) {{
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'author-result';
  if (summary.name === activeAuthorName) button.classList.add('active');
  button.dataset.authorName = summary.name;
  button.title = summary.name;

  const swatch = document.createElement('span');
  swatch.className = 'author-result-swatch';
  swatch.style.background = SCHOOL_COLORS[summary.primarySchool] || '#64748b';
  button.appendChild(swatch);

  const name = document.createElement('span');
  name.className = 'author-result-name';
  name.textContent = summary.name;
  button.appendChild(name);

  const meta = document.createElement('span');
  meta.className = 'author-result-meta';
  meta.textContent = `${{summary.count}}`;
  button.appendChild(meta);

  authorSearchResults.appendChild(button);
}}

function setActiveAuthor(authorName) {{
  activeAuthorName = authorName || '';
  Plotly.restyle('umap-plot', {{ 'marker.color': [getColors(getCurrentMode())], 'marker.size': [getMarkerSizes()] }}, [0]);
  renderAuthorResults();
}}

function applyColourState() {{
  const mode = modeSelect.value;
  if (selectedPointIndex !== null && !isDatumVisible(DATA[selectedPointIndex], mode)) {{
    selectedPointIndex = null;
    detailEl.textContent = 'Click a point to see its details.';
  }}
  Plotly.restyle('umap-plot', {{ 'marker.color': [getColors(mode)], 'marker.size': [getMarkerSizes()] }}, [0]);
  renderLegend(mode);
  renderAuthorResults();
  if (citationNetworkEnabled) {{
    drawCitationNetwork(selectedPointIndex !== null ? DATA[selectedPointIndex].paper_id : null);
  }} else {{
    clearEdges();
  }}
  if (selectedPointIndex !== null) {{
    renderDetail(DATA[selectedPointIndex]);
  }}
}}

function renderPaletteGrid() {{
  if (!activePaletteSchool) return;
  paletteGrid.innerHTML = '';
  const activeColor = SCHOOL_COLORS[activePaletteSchool];
  const defaultColor = DEFAULT_SCHOOL_COLORS[activePaletteSchool];
  paletteSubtitle.textContent = `${{activePaletteSchool}}. Colours are ordered by hue family, then increasing brightness.`;
  paletteCurrentSwatch.style.background = activeColor;
  paletteCurrentLabel.textContent = `Current: ${{activeColor}}  |  Default: ${{defaultColor}}`;

  SCHOOL_PICKER_COLORS.forEach((color, index) => {{
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'palette-chip';
    if (color === activeColor) button.classList.add('active');
    button.dataset.color = color;
    button.title = color;
    button.style.background = color;

    const label = document.createElement('span');
    label.className = 'palette-chip-label';
    label.textContent = String(index + 1);
    button.appendChild(label);

    paletteGrid.appendChild(button);
  }});
}}

function openPaletteForSchool(school) {{
  activePaletteSchool = school;
  renderPaletteGrid();
  paletteBackdrop.classList.add('open');
  paletteBackdrop.setAttribute('aria-hidden', 'false');
}}

function closePalette() {{
  paletteBackdrop.classList.remove('open');
  paletteBackdrop.setAttribute('aria-hidden', 'true');
  activePaletteSchool = null;
}}

function eventElementTarget(ev) {{
  return ev.target instanceof Element ? ev.target : ev.target?.parentElement || null;
}}

legendEl.addEventListener('change', ev => {{
  const groupToggle = eventElementTarget(ev)?.closest('[data-school-group-toggle]');
  if (groupToggle) {{
    setSchoolGroupVisible(groupToggle.dataset.schoolGroupToggle, groupToggle.checked);
  }}
}});

legendEl.addEventListener('click', ev => {{
  const toggle = eventElementTarget(ev)?.closest('[data-school-toggle]');
  if (toggle) {{
    const school = toggle.dataset.schoolToggle;
    if (hiddenSchools.has(school)) {{
      hiddenSchools.delete(school);
    }} else {{
      hiddenSchools.add(school);
    }}
    persistHiddenSchools();
    applyColourState();
    return;
  }}
  const collegeToggle = eventElementTarget(ev)?.closest('[data-college-toggle]');
  if (collegeToggle) {{
    const college = collegeToggle.dataset.collegeToggle;
    if (hiddenColleges.has(college)) {{
      hiddenColleges.delete(college);
    }} else {{
      hiddenColleges.add(college);
    }}
    persistHiddenColleges();
    applyColourState();
    return;
  }}
  const yearToggle = eventElementTarget(ev)?.closest('[data-year-toggle]');
  if (yearToggle) {{
    const year = yearToggle.dataset.yearToggle;
    if (hiddenYears.has(year)) {{
      hiddenYears.delete(year);
    }} else {{
      hiddenYears.add(year);
    }}
    persistHiddenYears();
    applyColourState();
    return;
  }}
  const button = eventElementTarget(ev)?.closest('.leg-label-btn');
  if (!button) return;
  openPaletteForSchool(button.dataset.school);
}});

paletteGrid.addEventListener('click', ev => {{
  const chip = eventElementTarget(ev)?.closest('.palette-chip');
  if (!chip || !activePaletteSchool) return;
  SCHOOL_COLORS[activePaletteSchool] = chip.dataset.color;
  persistSchoolColors();
  renderPaletteGrid();
  applyColourState();
}});

paletteClose.addEventListener('click', closePalette);
paletteBackdrop.addEventListener('click', ev => {{
  if (ev.target === paletteBackdrop) closePalette();
}});
paletteReset.addEventListener('click', () => {{
  if (!activePaletteSchool) return;
  SCHOOL_COLORS[activePaletteSchool] = DEFAULT_SCHOOL_COLORS[activePaletteSchool];
  persistSchoolColors();
  renderPaletteGrid();
  applyColourState();
}});
paletteResetAll.addEventListener('click', () => {{
  Object.keys(DEFAULT_SCHOOL_COLORS).forEach(school => {{
    SCHOOL_COLORS[school] = DEFAULT_SCHOOL_COLORS[school];
  }});
  persistSchoolColors();
  renderPaletteGrid();
  applyColourState();
}});
document.addEventListener('keydown', ev => {{
  if (ev.key === 'Escape' && paletteBackdrop.classList.contains('open')) closePalette();
}});

// ── colour mode switching ────────────────────────────────────────
const modeSelect = document.getElementById('colour-mode');
modeSelect.addEventListener('change', () => {{
  applyColourState();
}});

// ── UMAP projection switching ─────────────────────────────────────
const projectionSelect = document.getElementById('projection-select');

function switchProjection(projIdx) {{
  currentProjection = projIdx;
  const newXs = UMAP_PROJECTIONS.xs[projIdx];
  const newYs = UMAP_PROJECTIONS.ys[projIdx];
  // Keep DATA coords in sync (needed for edge rendering)
  DATA.forEach((d, i) => {{ d.x = newXs[i]; d.y = newYs[i]; }});
  Plotly.restyle('umap-plot', {{ x: [newXs], y: [newYs] }}, [0]);
  if (citationNetworkEnabled) {{
    drawCitationNetwork(selectedPointIndex !== null ? DATA[selectedPointIndex].paper_id : null);
  }} else if (selectedPointIndex !== null) {{
    drawSelectedEdges(DATA[selectedPointIndex].paper_id);
  }} else {{
    clearEdges();
  }}
}}

projectionSelect.addEventListener('change', () => {{
  switchProjection(parseInt(projectionSelect.value, 10));
}});

citationNetworkToggle.addEventListener('change', () => {{
  citationNetworkEnabled = citationNetworkToggle.checked;
  persistCitationNetworkEnabled();
  applyColourState();
}});

imagingOnlyToggle.addEventListener('change', () => {{
  imagingOnlyEnabled = imagingOnlyToggle.checked;
  persistImagingOnlyEnabled();
  applyColourState();
}});

pointSizeSlider.addEventListener('input', () => {{
  pointSize = parseInt(pointSizeSlider.value, 10);
  pointSizeValue.textContent = String(pointSize);
  persistPointSize();
  applyColourState();
}});

interactionModeSelect.addEventListener('change', () => {{
  interactionMode = interactionModeSelect.value === 'zoom' ? 'zoom' : 'pan';
  persistInteractionMode();
  Plotly.relayout('umap-plot', {{ dragmode: interactionMode }});
}});

authorSearchInput.addEventListener('input', renderAuthorResults);
authorSearchResults.addEventListener('click', ev => {{
  const button = eventElementTarget(ev)?.closest('.author-result');
  if (!button) return;
  setActiveAuthor(button.dataset.authorName);
}});
authorSearchReset.addEventListener('click', () => {{
  authorSearchInput.value = '';
  setActiveAuthor('');
}});
renderAuthorResults();

// ── panel toggling ───────────────────────────────────────────────
const panelToggle = document.getElementById('panel-toggle');
const panelPop = document.getElementById('panel-pop');
function setPanelHidden(h) {{
  document.body.classList.toggle('panel-hidden', h);
  panelToggle.innerHTML = h ? '&#9664; Show' : 'Hide &#9654;';
  setTimeout(() => Plotly.Plots.resize(document.getElementById('umap-plot')), 220);
}}
panelToggle.addEventListener('click', () => setPanelHidden(!document.body.classList.contains('panel-hidden')));
panelPop.addEventListener('click', () => setPanelHidden(false));

// ── detail rendering ─────────────────────────────────────────────
const detailEl = document.getElementById('paper-detail');
function esc(v) {{
  if (v == null) return '';
  return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function renderDetail(d) {{
  const schoolColor = SCHOOL_COLORS[d.school] || '#475569';
  const doi = d.doi ? `<a href="https://doi.org/${{esc(d.doi)}}" target="_blank" rel="noopener">${{esc(d.doi)}}</a>` : 'N/A';
  const oaLink = d.openalex_id
    ? `<a href="https://openalex.org/${{esc(d.openalex_id)}}" target="_blank" rel="noopener">${{esc(d.openalex_id)}}</a>`
    : 'N/A';
  const pmLink = d.pmid && d.pmid !== 'nan'
    ? `<a href="https://pubmed.ncbi.nlm.nih.gov/${{esc(d.pmid)}}/" target="_blank" rel="noopener">${{esc(d.pmid)}}</a>`
    : 'N/A';
  const nOut = (citesOut[d.paper_id] || []).length;
  const nIn  = (citedBy[d.paper_id] || []).length;

  detailEl.innerHTML = `
    <div class="paper-card" style="border-left-color:${{schoolColor}}">
      <h3>${{esc(d.title) || 'Untitled'}}</h3>
      <div class="meta-row"><strong>Year:</strong> ${{esc(d.year) || '?'}}</div>
      <div class="meta-row"><strong>Journal:</strong> ${{esc(d.journal) || '?'}}</div>
      <div class="meta-row"><strong>Authors:</strong> ${{esc(d.all_authors) || '?'}}</div>
      <div class="meta-row"><strong>Glasgow authors:</strong> ${{esc(d.glasgow_authors) || '?'}}</div>
      <div class="meta-row"><strong>School:</strong> <span style="color:${{schoolColor}};font-weight:600">${{esc(d.school) || '?'}}</span></div>
      <div class="meta-row"><strong>College:</strong> ${{esc(d.college) || '?'}}</div>
      <div class="meta-row"><strong>Total citations:</strong> ${{d.cited_by_count}}</div>
      <div class="meta-row"><strong>Cites in dataset:</strong> ${{nOut}} &nbsp; <strong>Cited by in dataset:</strong> ${{nIn}}</div>
      <div class="meta-row"><strong>OpenAlex:</strong> ${{oaLink}}</div>
      <div class="meta-row"><strong>PMID:</strong> ${{pmLink}}</div>
      <div class="meta-row"><strong>DOI:</strong> ${{doi}}</div>
      <div class="abstract-text">${{esc(d.abstract) || 'No abstract available.'}}</div>
    </div>`;
}}

// ── click + hover ────────────────────────────────────────────────
const plot = document.getElementById('umap-plot');

plot.on('plotly_click', ev => {{
  if (!ev || !ev.points || !ev.points.length) return;
  const i = ev.points[0].pointIndex;
  const d = DATA[i];
  if (!isDatumVisible(d, modeSelect.value)) return;
  selectedPointIndex = i;
  setPanelHidden(false);
  renderDetail(d);
  if (citationNetworkEnabled) {{
    drawCitationNetwork(d.paper_id);
  }} else {{
    clearEdges();
  }}
}});

plot.on('plotly_hover', ev => {{
  if (!ev || !ev.points || !ev.points.length) return;
  // lightweight highlight only; full detail on click
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
