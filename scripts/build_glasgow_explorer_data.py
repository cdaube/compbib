"""
Rebuild derived files for the Glasgow online explorer.

Reads:
  data/glasgow_abstracts.csv
  data/glasgow_authors.csv

Writes:
  data/glasgow_embeddings.npy
  data/glasgow_umap_coords.npy
  data/glasgow_umap_coords_multi.npy
  data/glasgow_umap_coords_3d_multi.npy
  glasgow_imaging_initiative_explorer.html
  glasgow_explorer.html  (legacy URL)

Usage:
    uv run python scripts/build_glasgow_explorer_data.py
    uv run python scripts/build_glasgow_explorer_data.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ABSTRACTS_CSV = os.path.join(DATA_DIR, "glasgow_abstracts.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "glasgow_embeddings.npy")
UMAP_FILE = os.path.join(DATA_DIR, "glasgow_umap_coords.npy")
UMAP_MULTI_FILE = os.path.join(DATA_DIR, "glasgow_umap_coords_multi.npy")
UMAP_3D_MULTI_FILE = os.path.join(DATA_DIR, "glasgow_umap_coords_3d_multi.npy")
MANIFEST_FILE = os.path.join(DATA_DIR, ".glasgow_explorer_data_manifest.json")

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_MAX_SEQ_LENGTH = 512
N_UMAP_RUNS = 10


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def embedding_texts(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        title = clean_text(row.get("title", ""))
        abstract = clean_text(row.get("abstract", ""))
        if abstract:
            text = f"search_document: {title} {abstract}"
        else:
            text = f"search_document: {title}"
        texts.append(text)
    return texts


def dataset_signature(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    cols = [c for c in ["paper_id", "pmid", "openalex_id", "title", "abstract"] if c in df.columns]
    for row in df[cols].fillna("").astype(str).itertuples(index=False, name=None):
        hasher.update("\t".join(row).encode("utf-8", errors="replace"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def load_manifest() -> dict[str, Any]:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict[str, Any]) -> None:
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def embeddings_are_current(df: pd.DataFrame, signature: str, force: bool) -> bool:
    if force or not os.path.exists(EMBEDDINGS_FILE):
        return False
    try:
        embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")
    except Exception:
        return False
    if embeddings.shape[0] != len(df):
        return False
    manifest = load_manifest()
    return manifest.get("dataset_signature") == signature and manifest.get("model_name") == MODEL_NAME


def choose_device(requested: str) -> str | None:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return None
    return None


def compute_embeddings(df: pd.DataFrame, device: str | None, batch_size: int) -> np.ndarray:
    print("Computing embeddings...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = MODEL_MAX_SEQ_LENGTH
    kwargs: dict[str, Any] = {
        "show_progress_bar": True,
        "batch_size": batch_size,
    }
    if device:
        kwargs["device"] = device
    embeddings = model.encode(embedding_texts(df), **kwargs)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved {EMBEDDINGS_FILE} {embeddings.shape}")
    return embeddings


def load_or_compute_embeddings(
    df: pd.DataFrame,
    signature: str,
    *,
    force: bool,
    device: str | None,
    batch_size: int,
) -> np.ndarray:
    if embeddings_are_current(df, signature, force):
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"Loaded current embeddings {embeddings.shape}")
        return embeddings
    embeddings = compute_embeddings(df, device=device, batch_size=batch_size)
    save_manifest(
        {
            "dataset_signature": signature,
            "n_papers": len(df),
            "model_name": MODEL_NAME,
            "model_max_seq_length": MODEL_MAX_SEQ_LENGTH,
        }
    )
    return embeddings


def umap_cache_is_current(path: str, expected_shape: tuple[int, ...], force: bool) -> bool:
    if force or not os.path.exists(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return tuple(arr.shape) == expected_shape


def compute_umaps(embeddings: np.ndarray, force: bool) -> None:
    n_papers = embeddings.shape[0]
    if (
        umap_cache_is_current(UMAP_FILE, (n_papers, 2), force)
        and umap_cache_is_current(UMAP_MULTI_FILE, (N_UMAP_RUNS, n_papers, 2), force)
        and umap_cache_is_current(UMAP_3D_MULTI_FILE, (N_UMAP_RUNS, n_papers, 3), force)
    ):
        print("Loaded current UMAP caches.")
        return

    if (
        umap_cache_is_current(UMAP_FILE, (n_papers, 2), force)
        and umap_cache_is_current(UMAP_MULTI_FILE, (N_UMAP_RUNS, n_papers, 2), force)
    ):
        print("Loaded current 2D UMAP caches.")
    else:
        runs = []
        for seed in range(N_UMAP_RUNS):
            print(f"Computing UMAP {seed + 1}/{N_UMAP_RUNS} (seed={seed})...")
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                n_components=2,
                random_state=seed,
            )
            runs.append(reducer.fit_transform(embeddings).astype(np.float32))

        multi = np.stack(runs, axis=0)
        np.save(UMAP_MULTI_FILE, multi)
        np.save(UMAP_FILE, multi[0])
        print(f"Saved {UMAP_FILE} {multi[0].shape}")
        print(f"Saved {UMAP_MULTI_FILE} {multi.shape}")

    if not umap_cache_is_current(UMAP_3D_MULTI_FILE, (N_UMAP_RUNS, n_papers, 3), force):
        runs_3d = []
        for seed in range(N_UMAP_RUNS):
            print(f"Computing 3D UMAP {seed + 1}/{N_UMAP_RUNS} (seed={seed})...")
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                n_components=3,
                random_state=seed,
            )
            runs_3d.append(reducer.fit_transform(embeddings).astype(np.float32))
        multi_3d = np.stack(runs_3d, axis=0)
        np.save(UMAP_3D_MULTI_FILE, multi_3d)
        print(f"Saved {UMAP_3D_MULTI_FILE} {multi_3d.shape}")


def build_html() -> None:
    print("Building explorer HTML...")
    subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "scripts", "make_glasgow_imaging_initiative_explorer.py")],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute embeddings and UMAPs even if caches look current.")
    parser.add_argument("--device", default="auto", help="Embedding device: auto, mps, cuda, cpu, etc.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-html", action="store_true", help="Only write embeddings/UMAP caches.")
    args = parser.parse_args()

    df = pd.read_csv(ABSTRACTS_CSV)
    signature = dataset_signature(df)
    device = choose_device(args.device)
    print(f"Loaded {len(df)} papers from {ABSTRACTS_CSV}")
    print(f"Embedding device: {device or 'sentence-transformers default'}")

    embeddings = load_or_compute_embeddings(
        df,
        signature,
        force=args.force,
        device=device,
        batch_size=args.batch_size,
    )
    compute_umaps(embeddings, force=args.force)
    if not args.skip_html:
        build_html()


if __name__ == "__main__":
    main()
