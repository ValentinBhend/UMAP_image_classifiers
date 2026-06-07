#!/usr/bin/env python3
"""
build_umap_site.py
==================
Turn saved feature vectors into an interactive, static UMAP-explorer website.

For every `<model>__<dataset>.npz` (features) that has a matching
`images__<dataset>.npz` (thumbnails), this builds a Bokeh scatter you can
pan / zoom, coloured by class, where HOVERING a point shows that image.
Outputs one standalone .html per combo + an index.html with a picker.

Interactive controls baked into every plot (no recompute, pure client-side):
  * point size slider
  * hover-image size slider
  * legend: click a class to show/hide ONLY that class

Optional --interactive adds n_neighbors and min_dist sliders. UMAP cannot run
in a browser, so this PRECOMPUTES a grid of embeddings (one per parameter
combination) offline and embeds them all; the sliders swap which one is shown.
Coordinates are stored as float32 (Bokeh's compact binary array format).

UMAP runs only here, offline. Results are cached as .npy in --cache-dir, which
is independent of the website and safe to gitignore. The website only ever
renders embedded coordinates + lazy-loaded thumbnails.

Two thumbnail modes (--mode):
  link  (default): shared docs/images/<dataset>/*.jpg, referenced by relative
                   path. Tiny HTML, lazy hover-loading, deduped across models.
                   Best for GitHub Pages.
  embed          : base64 the thumbnails into each .html -> one portable file.

Requirements:
    pip install umap-learn bokeh colorcet pillow numpy

Examples:
    python build_umap_site.py                                  # fast, single embedding
    python build_umap_site.py --interactive                    # + nn/min_dist sliders (slow)
    python build_umap_site.py --interactive --hp-neighbors 5,15,30,50,100 --hp-min-dist 0.0,0.1,0.25,0.5
    python build_umap_site.py --interactive --only resnet50__eurosat,vit_b16__cifar10
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
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap
from bokeh.models import (ColumnDataSource, HoverTool, CustomJS, Slider,
                          CDSView, GroupFilter)
from bokeh.palettes import Category10, Category20

try:
    import colorcet as cc
    _GLASBEY = list(cc.glasbey)
except Exception:
    _GLASBEY = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover(features_dir):
    combos, datasets = [], set()
    for path in sorted(glob.glob(os.path.join(features_dir, "*__*.npz"))):
        base = os.path.basename(path)[:-4]
        if base.startswith("images__"):
            continue
        model, _, dataset = base.partition("__")
        if os.path.exists(os.path.join(features_dir, f"images__{dataset}.npz")):
            combos.append((model, dataset))
            datasets.add(dataset)
        else:
            print(f"  ! no images__{dataset}.npz - skipping {base}")
    return combos, sorted(datasets)


# ---------------------------------------------------------------------------
# Feature loaders (lazy: only invoked on a UMAP cache miss)
#   - pretrained model: load the saved feature matrix
#   - raw pixels:        downsample the stored images, flatten, scale to [0,1]
# ---------------------------------------------------------------------------
RAW_PREFIX = "raw_"          # synthetic "models": raw_032px, raw_128px, ...


def raw_model_name(size):
    return f"raw_{size:03d}px"


def _downsample_flat(images, size):
    """(N,H,W,3) uint8 -> (N, size*size*3) float32 in [0,1]."""
    n, h, w, _ = images.shape
    if size == h == w:
        x = images.astype(np.float32)
    elif h % size == 0 and w % size == 0:                 # exact block-mean (fast)
        fh, fw = h // size, w // size
        x = images.reshape(n, size, fh, size, fw, 3).mean(axis=(2, 4))
    else:                                                  # arbitrary size (PIL)
        x = np.stack([np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS))
                      for a in images]).astype(np.float32)
    return (x.reshape(n, -1) / 255.0).astype(np.float32)


def make_feat_loader(model, dataset, features_dir, raw_size=None, indices=None):
    if model.startswith(RAW_PREFIX):
        def load():
            imgs = np.load(os.path.join(features_dir, f"images__{dataset}.npz"),
                           allow_pickle=True)["images"]
            if indices is not None:
                imgs = imgs[list(indices)]          # subset BEFORE downsample -> low peak RAM
            return _downsample_flat(imgs, raw_size)
        return load

    def load():
        feats = np.asarray(
            np.load(os.path.join(features_dir, f"{model}__{dataset}.npz"),
                    allow_pickle=True)["features"], dtype=np.float32)
        return feats if indices is None else feats[list(indices)]
    return load


def labels_for(model, dataset, features_dir, indices=None):
    src = f"images__{dataset}.npz" if model.startswith(RAW_PREFIX) else f"{model}__{dataset}.npz"
    data = np.load(os.path.join(features_dir, src), allow_pickle=True)
    lab = np.asarray(data["labels"]).astype(str)
    return lab if indices is None else lab[list(indices)]


def stored_image_size(dataset, features_dir):
    """Resolution at which images are stored (the raw-pixel ceiling)."""
    imgs = np.load(os.path.join(features_dir, f"images__{dataset}.npz"),
                   allow_pickle=True)["images"]
    return int(imgs.shape[1])


# ---------------------------------------------------------------------------
# UMAP grid (each (n_neighbors, min_dist) cached separately -> resumable)
# cache_model encodes raw resolution so different sizes never collide.
# ---------------------------------------------------------------------------
def grid_embeddings(cache_model, dataset, cache_dir, nn_vals, md_vals, metric, feat_loader,
                    low_memory="auto", tag_suffix=""):
    os.makedirs(cache_dir, exist_ok=True)
    feats = None
    out = {}
    for i, nn in enumerate(nn_vals):
        for j, md in enumerate(md_vals):
            tag = f"{cache_model}__{dataset}__nn{nn}_md{md}_{metric}{tag_suffix}"
            cache = os.path.join(cache_dir, tag + ".npy")
            if os.path.exists(cache):
                out[(i, j)] = np.load(cache)
                continue
            if feats is None:
                feats = feat_loader()
            # high-dim raw pixels are memory-hungry in UMAP's NN stage;
            # 'auto' turns low_memory on past 4096 dims (safe but slower).
            lowmem = (feats.shape[1] > 4096) if low_memory == "auto" else (low_memory == "on")
            print(f"      UMAP nn={nn} md={md} on {feats.shape}"
                  f"{' [low_memory]' if lowmem else ''} ...", end=" ", flush=True)
            emb = UMAP(n_neighbors=nn, min_dist=md, n_components=2, metric=metric,
                       low_memory=lowmem).fit_transform(feats) # , random_state=42
            np.save(cache, emb)
            out[(i, j)] = emb
            print("done")
    return out


# ---------------------------------------------------------------------------
# Thumbnails (one set per dataset, reused by every model)
# ---------------------------------------------------------------------------
def _jpeg(arr, thumb, quality):
    im = Image.fromarray(arr).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def thumbnail_refs(dataset, features_dir, out_dir, mode, thumb, quality, indices=None):
    """Return (refs, n). If indices is given, refs are aligned to that subset
    (refs[k] <-> indices[k]); link-mode files keep their ORIGINAL-index names so
    thumbnails stay shared/consistent across runs and sample sizes."""
    images = np.load(os.path.join(features_dir, f"images__{dataset}.npz"),
                     allow_pickle=True)["images"]
    sel = range(len(images)) if indices is None else list(indices)

    if mode == "embed":
        refs = ["data:image/jpeg;base64," +
                base64.b64encode(_jpeg(images[i], thumb, quality)).decode("ascii")
                for i in sel]
        return (refs, len(sel))

    img_dir = os.path.join(out_dir, "images", dataset)
    os.makedirs(img_dir, exist_ok=True)
    for i in sel:
        fp = os.path.join(img_dir, f"{i:05d}.jpg")
        if not os.path.exists(fp):
            with open(fp, "wb") as fh:
                fh.write(_jpeg(images[i], thumb, quality))
    return ([f"images/{dataset}/{i:05d}.jpg" for i in sel], len(sel))


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


def _tooltip(px):
    return ("""
    <div style="padding:5px">
      <img src="@img" style="width:var(--hover-thumb,%dpx);height:var(--hover-thumb,%dpx);
           display:block;margin:0 auto 5px;border:1px solid #d0d0d8;border-radius:4px;
           box-shadow:0 1px 4px rgba(0,0,0,.18);object-fit:cover"/>
      <div style="text-align:center;font:600 12px ui-sans-serif,system-ui">@label</div>
    </div>""" % (px, px))


# ---------------------------------------------------------------------------
# One plot (+ controls)
# ---------------------------------------------------------------------------
def build_plot(model, dataset, embs, nn_vals, md_vals, default_i, default_j,
               labels, refs, out_dir, display_px, point_size, resources="cdn"):
    classes = sorted(set(labels))
    interactive = (len(nn_vals) * len(md_vals)) > 1

    e0 = embs[(default_i, default_j)]
    src = ColumnDataSource(dict(
        x=e0[:, 0].astype(np.float32),
        y=e0[:, 1].astype(np.float32),
        label=list(map(str, labels)),
        img=list(refs),
    ))

    out_html = os.path.join(out_dir, f"{model}__{dataset}.html")
    output_file(out_html, title=f"{model} \u00d7 {dataset}", mode=resources)

    p = figure(
        sizing_mode="stretch_both", min_height=480, min_width=420,
        tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom",
        title=f"UMAP \u2014 {model} on {dataset}   "
              f"({len(labels)} images, {len(classes)} classes)",
        background_fill_color="#fbfbfd", border_fill_color="#fbfbfd",
    )

    palette = palette_for(classes)
    show_legend = len(classes) <= 20
    renderers = []
    if show_legend:
        # one renderer per class -> a legend click toggles ONLY that class
        for idx, cls in enumerate(classes):
            view = CDSView(filter=GroupFilter(column_name="label", group=cls))
            renderers.append(p.scatter("x", "y", source=src, view=view,
                                       size=point_size, alpha=0.7, line_color=None,
                                       color=palette[idx], legend_label=cls))
        p.add_layout(p.legend[0], "right")
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "9px"
        p.legend.title = "class (click to hide)"
        p.legend.title_text_font_style = "italic"
        p.legend.background_fill_alpha = 0.85
    else:
        renderers.append(p.scatter("x", "y", source=src, size=point_size, alpha=0.7,
                                   line_color=None,
                                   color=factor_cmap("label", palette=palette, factors=classes)))

    p.add_tools(HoverTool(renderers=renderers, tooltips=_tooltip(display_px),
                          point_policy="follow_mouse", attachment="vertical"))
    p.xaxis.visible = p.yaxis.visible = False
    p.xgrid.grid_line_color = p.ygrid.grid_line_color = None
    p.outline_line_color = None
    p.title.text_font_size = "11px"
    p.title.text_color = "#555"

    # ---- controls ----
    pt = Slider(start=1, end=16, value=point_size, step=1, title="point size", width=170)
    pt.js_on_change("value", CustomJS(args=dict(rs=renderers), code="""
        for (let i = 0; i < rs.length; i++) { rs[i].glyph.size = cb_obj.value; }
    """))

    th = Slider(start=48, end=256, value=display_px, step=8, title="hover image size", width=170)
    th.js_on_change("value", CustomJS(code="""
        document.documentElement.style.setProperty('--hover-thumb', cb_obj.value + 'px');
    """))

    controls = [pt, th]
    if interactive:
        grid_data = {}
        for (i, j), emb in embs.items():
            k = i * len(md_vals) + j
            grid_data["x_%d" % k] = emb[:, 0].astype(np.float32)
            grid_data["y_%d" % k] = emb[:, 1].astype(np.float32)
        grid_src = ColumnDataSource(grid_data)

        nn = Slider(start=0, end=len(nn_vals) - 1, value=default_i, step=1,
                    title="n_neighbors: %s" % nn_vals[default_i], width=230)
        md = Slider(start=0, end=len(md_vals) - 1, value=default_j, step=1,
                    title="min_dist: %s" % md_vals[default_j], width=230)
        hp = CustomJS(args=dict(src=src, grid=grid_src, nn=nn, md=md,
                                n_md=len(md_vals), nnv=list(nn_vals), mdv=list(md_vals)),
                      code="""
            const k = nn.value * n_md + md.value;
            const d = src.data;
            d['x'] = grid.data['x_' + k];
            d['y'] = grid.data['y_' + k];
            src.change.emit();
            nn.title = 'n_neighbors: ' + nnv[nn.value];
            md.title = 'min_dist: ' + mdv[md.value];
        """)
        nn.js_on_change("value", hp)
        md.js_on_change("value", hp)
        controls = [nn, md, pt, th]

    layout = column(row(*controls, sizing_mode="stretch_width", height=62),
                    p, sizing_mode="stretch_both")
    save(layout)
    return out_html


# ---------------------------------------------------------------------------
# Index page (only hand-written JS: ~15 lines swapping the iframe)
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
    font-weight:600; font-size:clamp(22px,3vw,34px); letter-spacing:.2px; color:var(--paper);
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
    background-size:5px 5px,5px 5px; background-repeat:no-repeat; transition:border-color .15s;
  }
  select:hover{border-color:var(--accent)}
  select:focus{outline:none; border-color:var(--accent2)}
  .stage{flex:1; position:relative; padding:18px 22px 22px}
  .frame{
    position:absolute; inset:18px 22px 22px; border:1px solid var(--line);
    border-radius:10px; overflow:hidden; background:#fbfbfd; box-shadow:0 10px 40px rgba(0,0,0,.45);
  }
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
</style>
</head><body>
<header>
  <div class="brand">
    <h1>UMAP image embeddings</h1>
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
  <b>drag</b> to pan &middot; sliders tune point / image size and UMAP params &middot;
  click a legend entry to hide a class.
</footer>

<script>
  // The ONLY hand-written JS: swap the iframe when a dropdown changes.
  const COMBOS = new Set(__COMBOS__);
  const m = document.getElementById('model');
  const d = document.getElementById('dataset');
  const f = document.getElementById('view');
  const msg = document.getElementById('msg');
  function update(){
    const key = m.value + '__' + d.value;
    if (COMBOS.has(key)){
      f.style.display = '';  msg.style.display = 'none';  f.src = key + '.html';
    } else {
      f.style.display = 'none';  msg.style.display = 'grid';
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
    ap.add_argument("--mode", choices=["link", "embed"], default="link")
    ap.add_argument("--thumb",   type=int, default=96,  help="stored thumbnail size (px)")
    ap.add_argument("--display", type=int, default=128, help="initial hover image size (px)")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--point-size", type=int, default=5, help="initial marker size")
    ap.add_argument("--neighbors", type=int,   default=15, help="single-embedding n_neighbors")
    ap.add_argument("--min-dist",  type=float, default=0.1, help="single-embedding min_dist")
    ap.add_argument("--metric",    default="cosine")
    ap.add_argument("--interactive", action="store_true",
                    help="add n_neighbors & min_dist sliders (precomputes a grid; slow, resumable)")
    ap.add_argument("--hp-neighbors", default="5,15,30,50,100",
                    help="grid of n_neighbors values when --interactive")
    ap.add_argument("--hp-min-dist",  default="0.0,0.1,0.25,0.5",
                    help="grid of min_dist values when --interactive")
    ap.add_argument("--only", default="", help="comma-separated model__dataset keys (default: all)")
    ap.add_argument("--raw-sizes", default="32,128",
                    help="raw-pixel resolutions to add as models, e.g. '32,128'. "
                         "Each becomes a model raw_<NNN>px. Capped at the stored image size.")
    ap.add_argument("--no-raw", action="store_true", help="do not add raw-pixel models")
    ap.add_argument("--low-memory", choices=["auto", "on", "off"], default="auto",
                    help="UMAP low_memory mode. auto = on when feature dim > 4096 "
                         "(memory-safe for raw_128px; slower). 'off' is faster if you have RAM.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap points per combo (0 = use all). Deterministic per dataset, so "
                         "all models share the same subset. Useful to make raw_128px fit RAM/time.")
    ap.add_argument("--title", default="pretrained features \u00b7 UMAP explorer")
    args = ap.parse_args()

    if args.interactive:
        nn_vals = [int(x) for x in args.hp_neighbors.split(",") if x.strip()]
        md_vals = [float(x) for x in args.hp_min_dist.split(",") if x.strip()]
        default_i = nn_vals.index(15) if 15 in nn_vals else len(nn_vals) // 2
        default_j = md_vals.index(0.1) if 0.1 in md_vals else len(md_vals) // 2
    else:
        nn_vals, md_vals, default_i, default_j = [args.neighbors], [args.min_dist], 0, 0

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, ".nojekyll"), "w").close()

    combos, datasets = discover(args.features_dir)

    # raw-pixel "models": one synthetic model per requested size, per dataset.
    # Reuse the same hover thumbnails; only the UMAP input differs.
    raw_size_of = {}                       # model name -> pixel size
    if not args.no_raw:
        req_sizes = sorted({int(x) for x in args.raw_sizes.split(",") if x.strip()})
        for dataset in datasets:
            ceil = stored_image_size(dataset, args.features_dir)
            for size in req_sizes:
                if size > ceil:
                    print(f"  ! raw {size}px > stored {ceil}px for {dataset} - skipping "
                          f"(re-extract at a larger THUMB_SIZE to enable)")
                    continue
                name = raw_model_name(size)
                raw_size_of[name] = size
                combos.append((name, dataset))

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        combos = [(m, d) for (m, d) in combos if f"{m}__{d}" in only]
    if not combos:
        print(f"No matching combos in {args.features_dir}")
        return

    n_grid = len(nn_vals) * len(md_vals)
    if args.interactive:
        mode_desc = f"interactive grid {len(nn_vals)}x{len(md_vals)} = {n_grid} embeddings/combo"
    else:
        mode_desc = "single embedding"
    print(f"{len(combos)} combo(s), mode={args.mode}, {mode_desc}")
    if args.interactive:
        print(f"  -> up to {n_grid * len(combos)} UMAP fits total (cached fits are skipped)\n")

    refs_cache = {}
    subset_cache = {}                                   # dataset -> indices (or None)
    suffix = f"_n{args.max_samples}" if args.max_samples else ""

    def subset_for(dataset):
        if dataset in subset_cache:
            return subset_cache[dataset]
        idx = None
        if args.max_samples:
            total = len(np.load(os.path.join(args.features_dir, f"images__{dataset}.npz"),
                                allow_pickle=True)["labels"])
            if total > args.max_samples:
                idx = np.sort(np.random.default_rng(42).choice(
                    total, args.max_samples, replace=False))
        subset_cache[dataset] = idx
        return idx

    for c, (model, dataset) in enumerate(combos, 1):
        print(f"[{c}/{len(combos)}] {model} \u00d7 {dataset}")
        idx = subset_for(dataset)
        if idx is not None and dataset not in refs_cache:
            print(f"      subsampling to {len(idx)} points (seed 42)")

        if dataset not in refs_cache:
            print(f"      thumbnails ({args.mode}) ...", end=" ", flush=True)
            refs_cache[dataset] = thumbnail_refs(dataset, args.features_dir, args.out_dir,
                                                 args.mode, args.thumb, args.quality, indices=idx)
            print(f"{refs_cache[dataset][1]} imgs")
        refs, n_img = refs_cache[dataset]

        raw_size    = raw_size_of.get(model)            # None for pretrained models
        feat_loader = make_feat_loader(model, dataset, args.features_dir, raw_size, indices=idx)
        embs   = grid_embeddings(model, dataset, args.cache_dir,
                                 nn_vals, md_vals, args.metric, feat_loader,
                                 low_memory=args.low_memory, tag_suffix=suffix)
        labels = labels_for(model, dataset, args.features_dir, indices=idx)
        n = min(len(labels), n_img, min(len(e) for e in embs.values()))
        if not (len(labels) == n_img == len(next(iter(embs.values())))):
            print(f"      ! length mismatch; aligning to {n}")
        embs = {k: v[:n] for k, v in embs.items()}

        out = build_plot(model, dataset, embs, nn_vals, md_vals, default_i, default_j,
                         labels[:n], refs[:n], args.out_dir, args.display, args.point_size,
                         resources="inline" if args.mode == "embed" else "cdn")
        print(f"      -> {out}")

    idx = build_index(combos, args.out_dir, args.title)
    print(f"\nWrote {idx}\nPush '{args.out_dir}' to GitHub Pages.")


if __name__ == "__main__":
    main()