"""
UMAP-of-UMAP, colored consistently with the field bar chart.

Caching: all network work (OpenAlex fields + Semantic Scholar embeddings,
joined on DOI) is saved to CACHE_FILE on the first run and loaded after.
Only UMAP + plot run every time. Delete CACHE_FILE (or change MAX_PAPERS /
the DOIs) to force a fresh pull.

>>> Strongly recommended: set S2_API_KEY. Without it you share a global
    5,000-requests-per-5-minutes pool and will hit 429s. A free key
    (https://www.semanticscholar.org/product/api) removes that problem.

Install:
    pip install requests numpy umap-learn matplotlib
"""

import os
import time
import requests
import numpy as np
import umap
import matplotlib.pyplot as plt

MAILTO = "you@example.com"                      # OpenAlex polite pool
S2_API_KEY = None                               # <-- put your free key here
UMAP_DOIS = ["10.48550/arXiv.1802.03426", "10.21105/joss.00861"]
MAX_PAPERS = 6000                               # cap for a clean plot
CACHE_FILE = "umap_citation_cache.npz"          # cached fetch (everything pre-UMAP)

OA = "https://api.openalex.org"
S2 = "https://api.semanticscholar.org/graph/v1"
S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
OA_SESSION = requests.Session(); OA_SESSION.params = {"mailto": MAILTO}


def norm_doi(doi):
    if not doi:
        return None
    return doi.lower().replace("https://doi.org/", "").strip()


def s2_post(url, params, payload, max_tries=8):
    """POST with exponential backoff. Raises a clear error if it can't recover,
    instead of returning a rate-limit body that later crashes mysteriously."""
    delay = 2.0
    last = None
    for attempt in range(max_tries):
        r = requests.post(url, params=params, json=payload, headers=S2_HEADERS)
        last = r
        if r.status_code == 200:
            data = r.json()
            if not isinstance(data, list):          # error object, not the paper list
                raise RuntimeError(f"Unexpected S2 response (not a list): {data}")
            return data
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After", delay))
            print(f"    S2 {r.status_code}; backing off {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_tries})")
            time.sleep(wait)
            delay = min(delay * 2, 60)              # exponential, capped
            continue
        r.raise_for_status()                        # other 4xx -> raise immediately
    # Out of retries: surface a real error rather than crashing downstream.
    raise RuntimeError(
        f"Semantic Scholar still failing after {max_tries} tries "
        f"(last status {last.status_code}). Set an API key to leave the "
        f"shared rate-limit pool."
    )


def resolve_oa_ids():
    ids = []
    for d in UMAP_DOIS:
        r = OA_SESSION.get(f"{OA}/works/doi:{d}")
        if r.status_code == 200:
            ids.append(r.json()["id"].rsplit("/", 1)[-1])
    return ids


def list_citing_with_field(oa_ids):
    """Return {doi: field_label} for citing works that have a DOI."""
    cites = "|".join(oa_ids)
    out, cursor = {}, "*"
    while cursor and len(out) < MAX_PAPERS:
        r = OA_SESSION.get(f"{OA}/works", params={
            "filter": f"cites:{cites}",
            "select": "doi,primary_topic",
            "per_page": 200,
            "cursor": cursor,
        })
        r.raise_for_status()
        data = r.json()
        for w in data["results"]:
            doi = norm_doi(w.get("doi"))
            if not doi:
                continue
            pt = w.get("primary_topic")
            field = pt["field"]["display_name"] if pt and pt.get("field") else "Unknown"
            out[doi] = field
        cursor = data["meta"].get("next_cursor")
        print(f"  collected {len(out)} citing works with DOI + field...")
        time.sleep(0.3)
    return out


def fetch_embeddings_by_doi(dois):
    """Batch-fetch SPECTER2 vectors; returns dict {doi: vector}."""
    vecs = {}
    for i in range(0, len(dois), 500):
        chunk = dois[i:i + 500]
        ids = [f"DOI:{d}" for d in chunk]
        papers = s2_post(f"{S2}/paper/batch",
                         params={"fields": "embedding.specter_v2"},
                         payload={"ids": ids})
        # batch preserves input order; missing papers come back as null
        for d, p in zip(chunk, papers):
            if isinstance(p, dict) and p.get("embedding") and "vector" in p["embedding"]:
                vecs[d] = p["embedding"]["vector"]
        print(f"  fetched embeddings for {len(vecs)} papers...")
        time.sleep(1.0 if S2_API_KEY else 2.0)     # gentler without a key
    return vecs


def build_dataset():
    """Do all the network work and return (dois, fields, X)."""
    print("Resolving UMAP records in OpenAlex ...")
    oa_ids = resolve_oa_ids()
    if not oa_ids:
        raise SystemExit("Could not resolve UMAP in OpenAlex.")

    print("Listing citing works + OpenAlex fields ...")
    doi_to_field = list_citing_with_field(oa_ids)

    print("Fetching SPECTER2 embeddings from Semantic Scholar ...")
    doi_to_vec = fetch_embeddings_by_doi(list(doi_to_field))

    dois = [d for d in doi_to_field if d in doi_to_vec]
    X = np.array([doi_to_vec[d] for d in dois], dtype=np.float32)
    fields = np.array([doi_to_field[d] for d in dois])
    return np.array(dois), fields, X


def load_or_build():
    """Load the joined dataset from cache, or build it and cache it."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached dataset from {CACHE_FILE} ...")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return d["dois"], d["fields"], d["X"]
    dois, fields, X = build_dataset()
    np.savez(CACHE_FILE, dois=dois, fields=fields, X=X)
    print(f"Saved dataset cache -> {CACHE_FILE}")
    return dois, fields, X


def main():
    dois, fields, X = load_or_build()
    print(f"Dataset: {X.shape} ({len(set(fields))} fields)")

    print("Running UMAP -> 2D ...")
    coords = umap.UMAP(n_neighbors=20, min_dist=0.2, metric="cosine",
                       random_state=42).fit_transform(X)

    uniq, counts = np.unique(fields, return_counts=True)
    top = set(uniq[np.argsort(counts)[::-1][:10]])
    display = np.array([f if f in top else "Other" for f in fields])

    plt.figure(figsize=(11, 9))
    for f in sorted(set(display)):
        m = display == f
        plt.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.6, label=f"{f} ({m.sum()})")
    plt.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.9)
    plt.title("UMAP citation landscape")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig("umap_colored_by_openalex_field.png", dpi=200)
    print("Saved -> umap_colored_by_openalex_field.png")


if __name__ == "__main__":
    main()