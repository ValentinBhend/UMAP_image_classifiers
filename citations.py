"""
Where is the UMAP paper used? — field/topic breakdown of citing papers.

Pulls every paper that cites UMAP from the OpenAlex API and aggregates them
by research field (and subfield) using OpenAlex's built-in classification.
No scraping, no download of individual papers — counting happens server-side.

Usage:
    pip install requests matplotlib
    python umap_citing_fields.py

Tip: put a real email in MAILTO. It puts you in OpenAlex's faster "polite pool"
and is just good manners. API keys are free but not required.
"""

import requests
import matplotlib.pyplot as plt

MAILTO = "you@example.com"          # <-- put your email here
BASE = "https://api.openalex.org"
SESSION = requests.Session()
SESSION.params = {"mailto": MAILTO}

# The UMAP paper exists as more than one record (the arXiv preprint and the
# JOSS publication). Citations are split across them, so we combine both for a
# complete picture. Unknown/expired DOIs are skipped automatically.
UMAP_DOIS = [
    "10.48550/arXiv.1802.03426",   # arXiv preprint (the heavily-cited one)
    "10.21105/joss.00861",         # Journal of Open Source Software version
]


def resolve_work_id(doi):
    r = SESSION.get(f"{BASE}/works/doi:{doi}")
    if r.status_code != 200:
        print(f"  (skipped {doi}: HTTP {r.status_code})")
        return None
    w = r.json()
    short_id = w["id"].rsplit("/", 1)[-1]
    print(f"  {short_id}  {w.get('cited_by_count', '?'):>6} citations  {w['title'][:60]}")
    return short_id


def group_citing_by(work_ids, dimension):
    """dimension is 'field', 'subfield', or 'domain'."""
    cites_filter = "|".join(work_ids)  # the pipe is a logical OR
    r = SESSION.get(
        f"{BASE}/works",
        params={
            "filter": f"cites:{cites_filter}",
            "group_by": f"primary_topic.{dimension}.id",
            "per_page": 200,
        },
    )
    r.raise_for_status()
    groups = r.json()["group_by"]
    # OpenAlex returns key_display_name + count, already sorted descending
    return [(g["key_display_name"], g["count"]) for g in groups if g["key_display_name"]]


def main():
    print("Resolving UMAP records:")
    work_ids = [wid for doi in UMAP_DOIS if (wid := resolve_work_id(doi))]
    if not work_ids:
        raise SystemExit("No UMAP records resolved — check the DOIs or your connection.")

    print("\nCiting papers by FIELD (26-category level):")
    fields = group_citing_by(work_ids, "field")
    total = sum(c for _, c in fields)
    for name, count in fields:
        print(f"  {count:>6}  ({count/total:5.1%})  {name}")

    # Headline chart: top 12 fields, horizontal bars (readable on a slide)
    top = fields[:12][::-1]
    labels = [n for n, _ in top]
    values = [c for _, c in top]
    plt.figure(figsize=(9, 6))
    plt.barh(labels, values, color="#4C9F70")
    plt.xlabel("Number of citing papers")
    plt.title("Where UMAP is used: citing papers by research field")
    for i, v in enumerate(values):
        plt.text(v, i, f" {v:,} ({v/total:.0%})", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("umap_citing_fields.png", dpi=200)
    print("\nSaved chart -> umap_citing_fields.png")

    # Optional finer cut for a backup/detail slide
    print("\nTop 15 SUBFIELDS:")
    for name, count in group_citing_by(work_ids, "subfield")[:15]:
        print(f"  {count:>6}  {name}")


if __name__ == "__main__":
    main()