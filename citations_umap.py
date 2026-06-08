"""
UMAP-of-UMAP: a 2D map of the papers that cite the UMAP paper.

Fixed pipeline (avoids the Semantic Scholar citations-endpoint quirks):
  1. List citing-paper IDs via /paper/{id}/citations  -- lightweight, IDs only.
     Uses the paper's CANONICAL S2 id, not the ARXIV: alias (the alias 400s here).
  2. Fetch SPECTER2 embeddings + field of study via POST /paper/batch
     -- the documented, reliable way to pull heavy fields for many papers.
  3. UMAP the 768-dim vectors to 2D and scatter-plot, colored by field.

Install:
    pip install requests numpy umap-learn matplotlib
"""

import time
import requests
import numpy as np
import umap
import matplotlib.pyplot as plt
import joblib
import os

# Canonical Semantic Scholar paperId for the UMAP paper (McInnes & Healy 2018).
# Using the hash avoids the ARXIV: alias that 400s on the citations endpoint.
UMAP_PAPER_ID = "3a288c63576fc385910cb5bc44eaea75b442e62e"

S2_API_KEY = None                 # optional: "your-key-here" for higher rate limits
MAX_PAPERS = 5000                 # plenty for a striking plot; endpoint caps near 10k
PAGE = 1000                       # citations page size
BATCH = 500                       # max ids per /paper/batch call

BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


def get(url, **kw):
    """GET with simple backoff on rate limits / transient 5xx."""
    for _ in range(6):
        r = requests.get(url, headers=HEADERS, **kw)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()


def post(url, **kw):
    for _ in range(6):
        r = requests.post(url, headers=HEADERS, **kw)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()


def list_citing_ids():
    """Page through citations, collecting just the citing-paper IDs."""
    ids, offset = [], 0
    while offset < MAX_PAPERS and offset + PAGE <= 10000:   # endpoint caps near 10k
        r = get(
            f"{BASE}/paper/{UMAP_PAPER_ID}/citations",
            params={"fields": "citingPaper.paperId", "offset": offset, "limit": PAGE},
        )
        batch = r.json().get("data", [])
        if not batch:
            break
        for row in batch:
            pid = (row.get("citingPaper") or {}).get("paperId")
            if pid:
                ids.append(pid)
        offset += PAGE
        print(f"  listed {len(ids)} citing-paper ids...")
        time.sleep(1)
    return ids


def fetch_embeddings(ids):
    """Batch-fetch SPECTER2 embeddings + primary field for each paper."""
    titles, fields, vectors = [], [], []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        r = post(
            f"{BASE}/paper/batch",
            params={"fields": "title,fieldsOfStudy,embedding.specter_v2"},
            json={"ids": chunk},
        )
        for p in r.json():
            if not p:
                continue
            emb = p.get("embedding")
            if not emb or "vector" not in emb:
                continue                       # no abstract -> no SPECTER2 vector
            vectors.append(emb["vector"])
            titles.append(p.get("title") or "")
            fos = p.get("fieldsOfStudy") or ["Unknown"]
            fields.append(fos[0])
        print(f"  fetched embeddings for {len(vectors)} papers...")
        time.sleep(1)
    return titles, np.array(fields), np.array(vectors, dtype=np.float32)


def main():
    print(f"Listing papers citing {UMAP_PAPER_ID} ...")
    if os.path.exists("list_citing_ids"):
        ids = joblib.load("list_citing_ids")
    else:
        ids = list_citing_ids()
        joblib.dump(ids, "list_citing_ids")
    print(f"Total citing ids: {len(ids)}")

    print("Fetching SPECTER2 embeddings via /paper/batch ...")
    if os.path.exists("fetch_embeddings"):
        titles, fields, X = joblib.load("fetch_embeddings")
    else:
        titles, fields, X = fetch_embeddings(ids)
        joblib.dump((titles, fields, X), "fetch_embeddings")
    print(f"Embedding matrix: {X.shape}")

    print("Running UMAP -> 2D ...")
    coords = umap.UMAP(
        n_neighbors=25, min_dist=0.1, metric="cosine", random_state=42
    ).fit_transform(X)

    uniq, counts = np.unique(fields, return_counts=True)
    top = set(uniq[np.argsort(counts)[::-1][:10]])
    display = np.array([f if f in top else "Other" for f in fields])

    plt.figure(figsize=(11, 9))
    for f in sorted(set(display)):
        m = display == f
        plt.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.6, label=f"{f} ({m.sum()})")
    plt.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.9)
    plt.title("The UMAP citation landscape (SPECTER2 embeddings -> UMAP 2D)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig("umap_of_umap_citations.png", dpi=200)
    print("Saved -> umap_of_umap_citations.png")

    np.savez("umap_citation_coords.npz",
             x=coords[:, 0], y=coords[:, 1], field=fields, title=np.array(titles))
    print("Saved -> umap_citation_coords.npz (x, y, field, title)")


if __name__ == "__main__":
    main()