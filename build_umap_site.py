#!/usr/bin/env python3
"""
build_umap_site.py
==================
Turn saved feature vectors into an interactive, static UMAP-explorer website.

For every `<model>__<dataset>.npz` (features) that has a matching
`images__<dataset>.npz` (thumbnails), this:
  1. computes a 2-D UMAP embedding (cached to disk -> re-runs are instant),
  2. builds a Bokeh scatter plot you can pan / zoom, coloured by class,
     where HOVERING a point shows that image,
  3. writes one standalone .html per combo + an index.html with a
     model / dataset picker.

No JavaScript to write yourself: Bokeh generates the plot's interactivity from
Python; the index page contains ~15 lines of trivial glue to swap the iframe.

Two ways to carry the thumbnails (choose with --mode):
  link  (default) : write a shared thumbnail folder (docs/images/<dataset>/*.jpg)
                    and reference it by relative path. HTML stays tiny, images
                    load lazily on hover, and they are shared across all models
                    of the same dataset. Best for GitHub Pages.
  embed           : base64 the thumbnails directly into each .html. Each file is
                    then fully self-contained / portable (one file = one demo),
                    at the cost of a larger file. Pass --mode embed.

Requirements:
    pip install umap-learn bokeh colorcet pillow numpy

Examples:
    # default: ./features  ->  ./docs   (GitHub-Pages-ready)
    python build_umap_site.py

    # one portable self-contained file per combo, smaller thumbs
    python build_umap_site.py --mode embed --thumb 80

    # tune UMAP
    python build_umap_site.py --neighbors 30 --min-dist 0.0 --metric cosine
"""

import os
import io
import glob
import json
import base64
import argparse

import numpy as np
from PIL import Image

from umap import UMAP
from bokeh.plotting import figure
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10, Category20

try:
    import colorcet as cc
    _GLASBEY = list(cc.glasbey)          # 256 maximally-distinct colours
except Exception:
    _GLASBEY = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover(features_dir):
    """Find (model, dataset) pairs that have BOTH a feature file and images."""
    combos, datasets = [], set()
    for path in sorted(glob.glob(os.path.join(features_dir, "*__*.npz"))):
        base = os.path.basename(path)[:-4]
        if base.startswith("images__"):                 # the image archives
            continue
        model, _, dataset = base.partition("__")         # split on FIRST "__"
        if os.path.exists(os.path.join(features_dir, f"images__{dataset}.npz")):
            combos.append((model, dataset))
            datasets.add(dataset)
        else:
            print(f"  ! no images__{dataset}.npz — skipping {base}")
    return combos, sorted(datasets)


# ---------------------------------------------------------------------------
# UMAP (cached)
# ---------------------------------------------------------------------------
def embedding_for(model, dataset, features_dir, cache_dir, umap_kw):
    os.makedirs(cache_dir, exist_ok=True)
    tag   = f"{model}__{dataset}__nn{umap_kw['n_neighbors']}_md{umap_kw['min_dist']}_{umap_kw['metric']}"
    cache = os.path.join(cache_dir, tag + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    data  = np.load(os.path.join(features_dir, f"{model}__{dataset}.npz"), allow_pickle=True)
    feats = np.asarray(data["features"], dtype=np.float32)
    print(f"      UMAP on {feats.shape} ...", end=" ", flush=True)
    emb = UMAP(**umap_kw).fit_transform(feats)
    np.save(cache, emb)
    print("done")
    return emb


def labels_for(model, dataset, features_dir):
    data = np.load(os.path.join(features_dir, f"{model}__{dataset}.npz"), allow_pickle=True)
    return np.asarray(data["labels"]).astype(str)


# ---------------------------------------------------------------------------
# Thumbnails  (one set per dataset, reused by every model)
# ---------------------------------------------------------------------------
def _jpeg(arr, thumb, quality):
    im = Image.fromarray(arr).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def thumbnail_refs(dataset, features_dir, out_dir, mode, thumb, quality):
    """Return a list of <img src> values (relative paths or data-URIs), aligned
    to the dataset's row order (== feature row order)."""
    images = np.load(os.path.join(features_dir, f"images__{dataset}.npz"),
                     allow_pickle=True)["images"]
    n = len(images)

    if mode == "embed":
        return ["data:image/jpeg;base64," + base64.b64encode(_jpeg(a, thumb, quality)).decode("ascii")
                for a in images], n

    # mode == "link": write files once, reference by relative path
    img_dir = os.path.join(out_dir, "images", dataset)
    existing = glob.glob(os.path.join(img_dir, "*.jpg"))
    if len(existing) != n:
        os.makedirs(img_dir, exist_ok=True)
        for i, a in enumerate(images):
            with open(os.path.join(img_dir, f"{i:05d}.jpg"), "wb") as fh:
                fh.write(_jpeg(a, thumb, quality))
    return [f"images/{dataset}/{i:05d}.jpg" for i in range(n)], n


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def palette_for(classes):
    n = len(classes)
    if n <= 2:
        return ["#4e79a7", "#f28e2b"][:n]
    if n <= 10:
        return list(Category10[max(3, n)])[:n]
    if n <= 20:
        return list(Category20[n])
    if _GLASBEY:
        return _GLASBEY[:n]
    base = list(Category20[20])
    return [base[i % 20] for i in range(n)]


# ---------------------------------------------------------------------------
# One plot
# ---------------------------------------------------------------------------
def build_plot(model, dataset, emb, labels, refs, out_dir, display_px, resources="cdn"):
    classes = sorted(set(labels))
    src = ColumnDataSource(dict(
        x=emb[:, 0].astype(float).tolist(),
        y=emb[:, 1].astype(float).tolist(),
        label=labels.tolist(),
        img=refs,
    ))

    out_html = os.path.join(out_dir, f"{model}__{dataset}.html")
    # resources="inline" embeds BokehJS in the file (works offline);
    # "cdn" keeps the file small but needs internet to load the library.
    output_file(out_html, title=f"{model} \u00d7 {dataset}", mode=resources)

    p = figure(
        sizing_mode="stretch_both",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        title=f"UMAP \u2014 {model} embeddings on {dataset}   "
              f"({len(labels)} images, {len(classes)} classes)",
        background_fill_color="#fbfbfd",
        border_fill_color="#fbfbfd",
    )

    show_legend = len(classes) <= 20
    glyph_kw = dict(
        size=5, alpha=0.7, line_color=None,
        color=factor_cmap("label", palette=palette_for(classes), factors=classes),
    )
    if show_legend:                          # pass legend_field ONLY when showing a legend;
        glyph_kw["legend_field"] = "label"   # Bokeh raises on legend_field=None
    p.scatter("x", "y", source=src, **glyph_kw)
    if show_legend:
        p.add_layout(p.legend[0], "right")
        p.legend.click_policy = "hide"          # click a class to toggle it
        p.legend.label_text_font_size = "9px"
        p.legend.title = "class  (click to hide)"
        p.legend.title_text_font_style = "italic"
        p.legend.background_fill_alpha = 0.85

    p.add_tools(HoverTool(
        tooltips=f"""
        <div style="padding:5px">
          <img src="@img" width="{display_px}" height="{display_px}"
               style="display:block;margin:0 auto 5px;border:1px solid #d0d0d8;
                      border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.18)"/>
          <div style="text-align:center;font:600 12px ui-sans-serif,system-ui">@label</div>
        </div>""",
        point_policy="follow_mouse",
        attachment="vertical",
    ))

    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.outline_line_color = None
    p.title.text_font_size = "11px"
    p.title.text_color = "#555"
    save(p)
    return out_html


# ---------------------------------------------------------------------------
# Index page  (the only place with hand-written JS: ~15 lines of iframe glue)
# ---------------------------------------------------------------------------
INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --ink:#0d1014; --panel:#12161c; --paper:#f4f1ea; --muted:#8b94a3;
    --line:#262d38; --accent:#d98c3f; --accent2:#5fb3b3;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; background:var(--ink); color:var(--paper);
    font-family:ui-monospace,"SF Mono",Menlo,"Cascadia Mono",monospace;
    background-image:radial-gradient(rgba(255,255,255,.035) 1px,transparent 1px);
    background-size:22px 22px;
    display:flex; flex-direction:column;
  }
  header{
    padding:26px 30px 18px; border-bottom:1px solid var(--line);
    display:flex; flex-wrap:wrap; align-items:flex-end; gap:8px 28px;
  }
  .brand{display:flex; flex-direction:column; gap:4px; margin-right:auto}
  h1{
    margin:0; font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    font-weight:600; font-size:clamp(22px,3vw,34px); letter-spacing:.2px;
    color:var(--paper);
  }
  h1 .em{color:var(--accent); font-style:italic}
  .sub{color:var(--muted); font-size:12px; letter-spacing:.14em; text-transform:uppercase}
  .controls{display:flex; flex-wrap:wrap; gap:18px}
  .field{display:flex; flex-direction:column; gap:6px}
  .field label{font-size:10px; letter-spacing:.18em; text-transform:uppercase; color:var(--muted)}
  select{
    appearance:none; background:var(--panel); color:var(--paper);
    border:1px solid var(--line); border-radius:7px; padding:9px 34px 9px 12px;
    font:inherit; font-size:13px; cursor:pointer; min-width:190px;
    background-image:linear-gradient(45deg,transparent 50%,var(--accent) 50%),
                     linear-gradient(135deg,var(--accent) 50%,transparent 50%);
    background-position:calc(100% - 17px) center,calc(100% - 12px) center;
    background-size:5px 5px,5px 5px; background-repeat:no-repeat;
    transition:border-color .15s;
  }
  select:hover{border-color:var(--accent)}
  select:focus{outline:none; border-color:var(--accent2)}
  .stage{flex:1; position:relative; padding:18px 22px 22px}
  .frame{
    position:absolute; inset:18px 22px 22px; border:1px solid var(--line);
    border-radius:10px; overflow:hidden; background:#fbfbfd;
    box-shadow:0 10px 40px rgba(0,0,0,.45);
  }
  /* corner ticks – instrument-panel detail */
  .frame::before,.frame::after{
    content:""; position:absolute; width:14px; height:14px; pointer-events:none;
    border-color:var(--accent); border-style:solid; z-index:3;
  }
  .frame::before{top:-1px;left:-1px;border-width:1px 0 0 1px}
  .frame::after{bottom:-1px;right:-1px;border-width:0 1px 1px 0}
  iframe{width:100%; height:100%; border:0; display:block; background:#fbfbfd}
  #msg{
    position:absolute; inset:0; display:none; place-items:center; text-align:center;
    color:#56607a; font-size:14px; padding:30px;
  }
  footer{padding:10px 30px 16px; color:var(--muted); font-size:11px; border-top:1px solid var(--line)}
  footer b{color:var(--accent2)}
  a{color:var(--accent2); text-decoration:none}
</style>
</head><body>
<header>
  <div class="brand">
    <h1>Embedding <span class="em">Atlas</span></h1>
    <div class="sub">__TITLE__</div>
  </div>
  <div class="controls">
    <div class="field">
      <label for="model">Pretrained model</label>
      <select id="model">__MODEL_OPTS__</select>
    </div>
    <div class="field">
      <label for="dataset">Test dataset</label>
      <select id="dataset">__DATASET_OPTS__</select>
    </div>
  </div>
</header>

<div class="stage">
  <div class="frame">
    <iframe id="view" title="UMAP plot"></iframe>
    <div id="msg"></div>
  </div>
</div>

<footer>
  UMAP of pretrained image-classifier features &middot;
  <b>hover</b> a point for its image &middot; <b>scroll</b> to zoom &middot;
  <b>drag</b> to pan &middot; click a legend entry to hide a class.
</footer>

<script>
  // The ONLY hand-written JS: swap the iframe when a dropdown changes.
  const COMBOS = new Set(__COMBOS__);          // "model__dataset" pairs that exist
  const m = document.getElementById('model');
  const d = document.getElementById('dataset');
  const f = document.getElementById('view');
  const msg = document.getElementById('msg');
  function update(){
    const key = m.value + '__' + d.value;
    if (COMBOS.has(key)){
      f.style.display = '';  msg.style.display = 'none';
      f.src = key + '.html';
    } else {
      f.style.display = 'none';
      msg.style.display = 'grid';
      msg.textContent = 'No precomputed plot for ' + m.value + ' \u00d7 ' + d.value + '.';
    }
  }
  m.addEventListener('change', update);
  d.addEventListener('change', update);
  update();
</script>
</body></html>
"""


def build_index(combos, out_dir, title):
    models   = sorted({m for m, _ in combos})
    datasets = sorted({d for _, d in combos})
    combo_keys = [f"{m}__{d}" for m, d in combos]

    def opts(values):
        return "".join(f'<option value="{v}">{v}</option>' for v in values)

    html = (INDEX_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__MODEL_OPTS__", opts(models))
            .replace("__DATASET_OPTS__", opts(datasets))
            .replace("__COMBOS__", json.dumps(combo_keys)))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default="./features")
    ap.add_argument("--out-dir",      default="./docs")
    ap.add_argument("--cache-dir",    default="./umap_cache")
    ap.add_argument("--mode", choices=["link", "embed"], default="link",
                    help="link = shared thumbnail folder (small HTML, GitHub-Pages-ready); "
                         "embed = data-URIs baked into each self-contained file")
    ap.add_argument("--thumb",   type=int, default=96,  help="stored thumbnail size (px)")
    ap.add_argument("--display", type=int, default=128, help="hover image size (px)")
    ap.add_argument("--quality", type=int, default=80,  help="JPEG quality")
    ap.add_argument("--neighbors", type=int,   default=15)
    ap.add_argument("--min-dist",  type=float, default=0.1)
    ap.add_argument("--metric",    default="cosine",
                    help="UMAP metric; 'cosine' suits deep features, 'euclidean' is the UMAP default")
    ap.add_argument("--title", default="pretrained features \u00b7 UMAP explorer")
    args = ap.parse_args()

    umap_kw = dict(n_neighbors=args.neighbors, min_dist=args.min_dist,
                   n_components=2, metric=args.metric, random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, ".nojekyll"), "w").close()   # let GH Pages serve as-is

    combos, datasets = discover(args.features_dir)
    if not combos:
        print(f"No <model>__<dataset>.npz (+ images__<dataset>.npz) pairs in {args.features_dir}")
        return
    print(f"Found {len(combos)} combo(s) across {len(datasets)} dataset(s). "
          f"mode={args.mode}\n")

    refs_cache = {}   # dataset -> (refs, n)   built once, reused across models
    for i, (model, dataset) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {model} \u00d7 {dataset}")

        if dataset not in refs_cache:
            print(f"      thumbnails ({args.mode}) ...", end=" ", flush=True)
            refs_cache[dataset] = thumbnail_refs(
                dataset, args.features_dir, args.out_dir,
                args.mode, args.thumb, args.quality)
            print(f"{refs_cache[dataset][1]} imgs")
        refs, n_img = refs_cache[dataset]

        emb    = embedding_for(model, dataset, args.features_dir, args.cache_dir, umap_kw)
        labels = labels_for(model, dataset, args.features_dir)

        n = min(len(emb), len(labels), n_img)
        if not (len(emb) == len(labels) == n_img):
            print(f"      ! length mismatch (emb={len(emb)}, labels={len(labels)}, "
                  f"imgs={n_img}); aligning to {n}")
        out = build_plot(model, dataset, emb[:n], labels[:n], refs[:n],
                         args.out_dir, args.display,
                         resources="inline" if args.mode == "embed" else "cdn")
        print(f"      -> {out}")

    idx = build_index(combos, args.out_dir, args.title)
    print(f"\nWrote {idx}")
    print(f"Open it locally, or push '{args.out_dir}' to GitHub Pages.")


if __name__ == "__main__":
    main()