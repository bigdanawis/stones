"""
End-to-end demo: load 10 WRL files → clean → sample → embed → PCA plot.

Outputs written to:  outputs/demo/
  embeddings/<stem>.npy      per-object embedding  [D]
  pointclouds/<stem>.npy     sampled point cloud   [N, 6]
  metadata.csv               geometry + path info
  failed_meshes.csv          any files that failed
  pca.png                    2-D PCA scatter

Usage
-----
  python scripts/demo_embed_pca.py
  python scripts/demo_embed_pca.py --n 20 --sampling_mode uniform
  python scripts/demo_embed_pca.py --skip_clean --decimate --target_ratio 0.02
"""
import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_cleaning import clean_mesh, compute_surface_area
from src.mesh_io import load_mesh
from src.pointmae_embedder import build_embedder
from src.sampling import sample_points
from src.utils import (
    FailureWriter, MetadataWriter, Timer,
    collect_mesh_files, ensure_dir, get_logger,
)

log = get_logger("demo")

WRL_DIR    = ROOT / "wrl"
OUTPUT_DIR = ROOT / "outputs" / "demo"


# ---------------------------------------------------------------------------
# Normalize helpers (same as extract script)
# ---------------------------------------------------------------------------

def normalize_mesh(vertices):
    centroid = vertices.mean(axis=0)
    v = vertices - centroid
    bbox = v.max(axis=0) - v.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    scale = diag if diag > 0 else 1.0
    return v / scale, centroid, scale, bbox


# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------

def process_one(path, embedder, args, out, meta_writer, fail_writer, rng):
    stem = path.stem
    timer = Timer()

    # 1. Load
    try:
        raw = load_mesh(path)
    except Exception as e:
        log.error(f"  LOAD FAILED: {e}")
        fail_writer.write(path.name, "load", str(e))
        return None

    v, f = raw["vertices"], raw["faces"]
    log.info(f"  loaded  {len(v):>8,} V  {len(f):>8,} F  via {raw['source']}")

    # 2. Clean (optional)
    if args.skip_clean:
        vc, fc = v, f
        clean_meta = {"num_clean_vertices": len(v), "num_clean_faces": len(f)}
    else:
        try:
            vc, fc, clean_meta = clean_mesh(v, f, keep_largest_component=True)
        except Exception as e:
            log.error(f"  CLEAN FAILED: {e}")
            fail_writer.write(path.name, "clean", str(e))
            return None
        if len(fc) == 0:
            fail_writer.write(path.name, "clean", "0 faces after cleaning")
            return None

    # 3. Decimate (optional)
    if args.decimate:
        from src.decimation import decimate
        from src.mesh_cleaning import orient_normals_outward
        tf = args.target_faces or max(500, int(len(fc) * args.target_ratio))
        try:
            vc, fc = decimate(vc, fc, tf, method=args.decimate_method)
            fc = orient_normals_outward(vc, fc)
        except Exception as e:
            log.warning(f"  DECIMATE FAILED ({e}), continuing with full mesh")

    log.info(f"  clean   {len(vc):>8,} V  {len(fc):>8,} F")

    # 3b. Projection PNG + axis-aligned dimensions
    extents = visualize_mesh_projections(stem, vc, fc, out)
    log.info(f"  dims    length={extents[0]:.3f}  width={extents[1]:.3f}  depth={extents[2]:.3f}"
             f"  (principal-axis aligned, mesh units)")

    # 4. Normalize
    vn, centroid, scale, bbox = normalize_mesh(vc)
    surface_area = compute_surface_area(vc, fc)

    # 5. Sample
    try:
        pts, nrm = sample_points(vn, fc, args.num_points,
                                 mode=args.sampling_mode, rng=rng)
    except Exception as e:
        log.error(f"  SAMPLE FAILED: {e}")
        fail_writer.write(path.name, "sample", str(e))
        return None

    # 6. Embed
    try:
        embedding = embedder.embed(pts)
    except Exception as e:
        log.error(f"  EMBED FAILED: {e}")
        fail_writer.write(path.name, "embed", str(e))
        return None

    # 7. Save
    pc_path  = out / "pointclouds" / f"{stem}.npy"
    emb_path = out / "embeddings"  / f"{stem}.npy"
    ensure_dir(pc_path.parent)
    ensure_dir(emb_path.parent)
    np.save(pc_path,  np.concatenate([pts, nrm], axis=1))
    np.save(emb_path, embedding)
    np.savetxt(emb_path.with_suffix(".csv"), embedding[np.newaxis], delimiter=",")

    log.info(f"  embed   dim={len(embedding)}  elapsed={timer.elapsed():.1f}s")

    meta_writer.write({
        "filename":              path.name,
        "num_original_vertices": len(v),
        "num_original_faces":    len(f),
        "num_clean_vertices":    clean_meta["num_clean_vertices"],
        "num_clean_faces":       clean_meta["num_clean_faces"],
        "sampling_mode":         args.sampling_mode,
        "num_points":            args.num_points,
        "centroid_x": centroid[0], "centroid_y": centroid[1], "centroid_z": centroid[2],
        "scale_factor":          scale,
        "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_z": bbox[2],
        # SVD-aligned extents (principal-axis bounding box)
        "dim_length": float(extents[0]),
        "dim_width":  float(extents[1]),
        "dim_depth":  float(extents[2]),
        "surface_area":          surface_area,
        "embedding_path":        str(emb_path),
        "pointcloud_path":       str(pc_path),
    })
    return stem, embedding, extents


# ---------------------------------------------------------------------------
# Site map from Excel
# ---------------------------------------------------------------------------

XLSX_DEFAULT = ROOT / "wrl" / "Handaxes 2026 list with sites.xlsx"


_TOOL_ABBREV = {
    "handaxe":   "h",
    "cleaver":   "c",
    "discoid":   "d",
    "dicoid":    "d",   # common typo in the sheet
    "trihedral": "t",
    "pick":      "p",
    "scraper":   "s",
    "scaper":    "s",   # variant/typo (sidescaper)
}


def _parse_note(raw) -> str | None:
    """
    Convert a raw notes cell to a short annotation string, or None.

    Rules:
      "on a flake"                       -> "f"
      known tool type(s) [with /]        -> abbreviation(s) e.g. "h", "c/d", "t/p"
      any other non-empty comment        -> "-"
      empty / NaN                        -> None (no annotation)
    """
    if raw is None:
        return None
    note = str(raw).strip()
    if note in ("", "nan", "NaN", "None"):
        return None

    note_lc = note.lower()

    if "on a flake" in note_lc:
        return "f"

    # Split on "/" to handle combos like "cleaver/ discoid?"
    parts = [p.strip().rstrip("?").strip().lower() for p in note_lc.split("/")]
    abbrevs = []
    for part in parts:
        matched = None
        for kw, ab in _TOOL_ABBREV.items():
            if kw in part:
                matched = ab
                break
        abbrevs.append(matched)

    if abbrevs and all(a is not None for a in abbrevs):
        # Deduplicate while preserving order
        seen, deduped = set(), []
        for a in abbrevs:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        return "/".join(deduped)

    # Has content but doesn't fully match known tool types
    return "-"


def load_site_map(xlsx_path: Path) -> tuple:
    """
    Read the Excel spreadsheet and return (site_map, notes_map).

    site_map  : {wrl_stem: site_name}
    notes_map : {wrl_stem: annotation_str}  — None values excluded

    Layout:
      Row 0  — site headers in every odd column (1, 3, 5, …)
      Row 1+ — specimen names (WRL stems) under each site column
      Even columns (col+1) are notes for that site's specimens.

    Site name is taken as the text before the first '(' (stripped).
    """
    import pandas as pd

    df = pd.read_excel(xlsx_path, header=None)
    site_cols = [c for c in df.columns if c % 2 == 1]   # cols 1, 3, 5, …

    site_map  = {}
    notes_map = {}

    for col in site_cols:
        raw_site = str(df.iloc[0, col])
        site = raw_site.split("(")[0].strip().rstrip(" -_")
        notes_col = col + 1   # adjacent even column holds notes

        has_notes = notes_col in df.columns

        for row_idx in range(1, len(df)):
            val = df.iloc[row_idx, col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            stem = str(val).strip()
            site_map[stem] = site

            if has_notes:
                note_val = df.iloc[row_idx, notes_col]
                ann = _parse_note(note_val)
                if ann is not None:
                    notes_map[stem] = ann

    log.info(f"Site map : {len(site_map)} specimens across "
             f"{len(set(site_map.values()))} sites")
    log.info(f"Notes map: {len(notes_map)} annotated specimens "
             f"({', '.join(sorted(set(notes_map.values())))})")
    return site_map, notes_map


# ---------------------------------------------------------------------------
# 2-D mesh projection (3 planes in principal-axis frame)
# ---------------------------------------------------------------------------

def visualize_mesh_projections(stem, vertices, faces, out_dir, max_faces=6000):
    """
    Project the mesh onto its 3 principal planes and save a 3-panel PNG.

    Axes are found via SVD of the centred vertex cloud:
      axis-0  longest extent  (major / length)
      axis-1  widest extent   (intermediate / width)
      axis-2  shortest extent (depth)

    Panels:
      left   axis-0 vs axis-1  (top view  — length x width)
      centre axis-0 vs axis-2  (side view — length x depth)
      right  axis-1 vs axis-2  (front view— width  x depth)

    Shading: diffuse from a fixed 3-D light applied to the 3-D face normals,
    giving consistent shape cues across all three views.
    Edges: thin semi-transparent lines drawn on top of filled faces.
    """
    from matplotlib.collections import PolyCollection, LineCollection

    # -- 1. Find principal axes via SVD --
    centred = vertices - vertices.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)  # rows = principal axes
    projected = centred @ Vt.T                              # [V, 3] in aligned frame

    # -- 2. Subsample faces --
    rng = np.random.default_rng(0)
    if len(faces) > max_faces:
        sel = rng.choice(len(faces), max_faces, replace=False)
        faces_draw = faces[sel]
    else:
        faces_draw = np.asarray(faces)

    tri_verts = projected[faces_draw]                       # [F, 3, 3]

    # -- 3. Compute per-face normals in aligned frame and diffuse shading --
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    raw_n = np.cross(v1 - v0, v2 - v0)                     # [F, 3] unnormalised
    nlen  = np.linalg.norm(raw_n, axis=1, keepdims=True)
    safe  = nlen[:, 0] > 0
    normals = np.zeros_like(raw_n)
    normals[safe] = raw_n[safe] / nlen[safe]

    # Per-face centroid-based orientation fix: flip normals pointing inward.
    # In the SVD-aligned frame the mesh centroid is exactly zero, so the
    # outward direction for each face is simply its centroid vector.
    face_centroids = (v0 + v1 + v2) / 3.0          # [F, 3]
    inward = np.einsum("fi,fi->f", normals, face_centroids) < 0
    normals[inward] = -normals[inward]

    # Fixed light slightly above and to the upper-right in aligned space
    light = np.array([0.4, 0.3, 0.85])
    light /= np.linalg.norm(light)
    diffuse = np.clip(normals @ light, 0, 1)               # [F]

    # Base colour: warm stone grey; ambient + diffuse
    ambient  = 0.30
    strength = 0.65
    lum = ambient + strength * diffuse                      # [F]  in [0.30, 0.95]
    # Stone-warm tint
    r = np.clip(lum * 0.82, 0, 1)
    g = np.clip(lum * 0.78, 0, 1)
    b = np.clip(lum * 0.72, 0, 1)
    face_colors = np.stack([r, g, b, np.ones(len(lum))], axis=1)  # [F, 4] RGBA

    # Edge colour: slightly lighter than the darkest face, semi-transparent
    edge_rgba = (0.55, 0.53, 0.50, 0.22)

    # -- 4. Extents for labels --
    extents = projected.max(axis=0) - projected.min(axis=0)
    axis_labels = [
        f"length ({extents[0]:.1f})",
        f"width  ({extents[1]:.1f})",
        f"depth  ({extents[2]:.1f})",
    ]

    panels = [
        (0, 1, "top view",   axis_labels[0], axis_labels[1]),
        (0, 2, "side view",  axis_labels[0], axis_labels[2]),
        (1, 2, "front view", axis_labels[1], axis_labels[2]),
    ]

    # -- 5. Draw --
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#111111")
    fig.suptitle(stem, color="white", fontsize=11, y=1.01)

    pad = 0.04
    for ax, (xi, yi, view, xl, yl) in zip(axes, panels):
        tris_2d = tri_verts[:, :, [xi, yi]]                 # [F, 3, 2]

        # Filled faces with diffuse shading
        faces_col = PolyCollection(
            tris_2d,
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0,
            zorder=1,
        )
        ax.add_collection(faces_col)

        # Edge lines: extract the 3 edges of every triangle as line segments
        # Shape: [F*3, 2, 2] — each row is one edge (start, end) in 2D
        segs = np.concatenate([
            tris_2d[:, [0, 1]],
            tris_2d[:, [1, 2]],
            tris_2d[:, [2, 0]],
        ], axis=0)                                           # [F*3, 2, 2]
        edge_col = LineCollection(
            segs,
            colors=[edge_rgba],
            linewidths=0.35,
            zorder=2,
        )
        ax.add_collection(edge_col)

        xs = projected[:, xi]
        ys = projected[:, yi]
        xr = xs.max() - xs.min()
        yr = ys.max() - ys.min()
        ax.set_xlim(xs.min() - xr * pad, xs.max() + xr * pad)
        ax.set_ylim(ys.min() - yr * pad, ys.max() + yr * pad)
        ax.set_aspect("equal")

        ax.set_facecolor("#111111")
        ax.set_title(view, color="#aaaaaa", fontsize=9, pad=4)
        ax.set_xlabel(xl, color="#777777", fontsize=7)
        ax.set_ylabel(yl, color="#777777", fontsize=7)
        ax.tick_params(colors="#555555", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    plt.tight_layout()
    png_dir = out_dir / "projections"
    ensure_dir(png_dir)
    out_path = png_dir / f"{stem}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    return extents   # [width_along_major, width_along_inter, width_along_depth]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Colorblind-safe palette — combines Paul Tol's "bright" and Wong (2011) schemes.
# Distinguishable under deuteranopia, protanopia and tritanopia; readable on
# both dark and light backgrounds.
_PALETTE = [
    "#4477AA",  # blue        (Tol bright)
    "#EE6677",  # red         (Tol bright)
    "#117722",  # green       (Tol bright)
    "#DDCC55",  # yellow      (Tol bright)
    "#66CCEE",  # cyan        (Tol bright)
    "#992266",  # purple      (Tol bright)
    "#E69F00",  # orange      (Wong 2011)
    "#00AE83",  # teal        (Wong 2011)
    "#D55E00",  # vermillion  (Wong 2011)
    "#DD89B7",  # pink        (Wong 2011)
    "#0072B2",  # dark blue   (Wong 2011)
    "#44AA99",  # teal-green  (Tol muted)
    "#999933",  # olive       (Tol muted)
    "#882255",  # wine        (Tol muted)
    "#332288",  # indigo      (Tol muted)
]




def _scale_and_reduce(embeddings, n_components):
    """StandardScale then PCA. Returns (coords, explained_variance_ratio, E_scaled)."""
    E = np.stack(embeddings)
    E_scaled = StandardScaler().fit_transform(E)
    n = min(len(embeddings), E.shape[1], n_components)
    pca = PCA(n_components=n)
    coords = pca.fit_transform(E_scaled)
    # Pad to requested dimensionality if fewer components available
    while coords.shape[1] < n_components:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])
    var = list(pca.explained_variance_ratio_)
    while len(var) < n_components:
        var.append(0.0)
    return coords, var, E_scaled


def _site_colors(stems, site_map):
    """Return (stem_sites, unique_sites, site_color_dict)."""
    stem_sites   = [site_map.get(s, "Unknown") for s in stems] if site_map else ["All"] * len(stems)
    unique_sites = sorted(set(stem_sites))
    site_color   = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(unique_sites)}
    return stem_sites, unique_sites, site_color


def _plotly_scalar_3d(stems, coords3, values, axis_labels, title, notes_map=None):
    """Plotly 3-D scatter coloured by a continuous scalar (single trace + colorbar)."""
    import plotly.graph_objects as go

    vals  = np.array(values, dtype=float)
    ann   = [notes_map.get(s, "") if notes_map else "" for s in stems]
    hover = [
        f"<b>{s}</b><br>value: {v:.4f}" + (f"<br>type: {a}" if a else "")
        for s, v, a in zip(stems, vals, ann)
    ]

    trace = go.Scatter3d(
        x=coords3[:, 0], y=coords3[:, 1], z=coords3[:, 2],
        mode="markers+text",
        marker=dict(
            size=6,
            color=vals,
            colorscale="Plasma",
            colorbar=dict(
                title=dict(text=title.split("—")[-1].strip(),
                           font=dict(color="#aaaaaa")),
                tickfont=dict(color="#aaaaaa"),
            ),
            opacity=0.85,
            line=dict(color="white", width=0.5),
        ),
        text=ann,
        textposition="top center",
        textfont=dict(size=8, color="white"),
        hovertext=hover,
        hoverinfo="text",
    )

    dark = "#0f0f1a"
    layout = go.Layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        paper_bgcolor=dark,
        scene=dict(
            xaxis=dict(title=axis_labels[0], color="#aaaaaa",
                       gridcolor="#333333", backgroundcolor=dark),
            yaxis=dict(title=axis_labels[1], color="#aaaaaa",
                       gridcolor="#333333", backgroundcolor=dark),
            zaxis=dict(title=axis_labels[2], color="#aaaaaa",
                       gridcolor="#333333", backgroundcolor=dark),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return go.Figure(data=[trace], layout=layout)


def _plotly_3d(stems, coords3, axis_labels, title, site_map, notes_map):
    """Build a plotly Figure with a 3-D scatter coloured by site."""
    import plotly.graph_objects as go

    stem_sites, unique_sites, site_color = _site_colors(stems, site_map)

    traces = []
    for site in unique_sites:
        idxs = [i for i, s in enumerate(stem_sites) if s == site]
        ann  = [notes_map.get(stems[i], "") if notes_map else "" for i in idxs]
        hover = [
            f"<b>{stems[i]}</b><br>site: {site}" + (f"<br>type: {ann[k]}" if ann[k] else "")
            for k, i in enumerate(idxs)
        ]
        traces.append(go.Scatter3d(
            x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
            mode="markers+text",
            name=site,
            marker=dict(size=6, color=site_color[site], opacity=0.85,
                        line=dict(color="white", width=0.5)),
            text=ann,
            textposition="top center",
            textfont=dict(size=8, color="white"),
            hovertext=hover,
            hoverinfo="text",
        ))

    dark = "#0f0f1a"
    layout = go.Layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        paper_bgcolor=dark, plot_bgcolor=dark,
        scene=dict(
            xaxis=dict(title=axis_labels[0], color="#aaaaaa",
                       gridcolor="#333333", backgroundcolor=dark),
            yaxis=dict(title=axis_labels[1], color="#aaaaaa",
                       gridcolor="#333333", backgroundcolor=dark),
            zaxis=dict(title=axis_labels[2], color="#aaaaaa",
                       gridcolor="#333333", backgroundcolor=dark),
        ),
        legend=dict(font=dict(color="white"), bgcolor="#1a1a2e",
                    bordercolor="#444444", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return go.Figure(data=traces, layout=layout)


# ---------------------------------------------------------------------------
# PCA plot
# ---------------------------------------------------------------------------

def plot_pca(stems, embeddings, out, site_map=None, notes_map=None):
    coords, var, _ = _scale_and_reduce(embeddings, n_components=10)
    stem_sites, unique_sites, site_color = _site_colors(stems, site_map)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    if site_map:
        for site in unique_sites:
            idxs = [i for i, s in enumerate(stem_sites) if s == site]
            ax.scatter(coords[idxs, 0], coords[idxs, 1],
                       color=site_color[site], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5, label=site)

        legend = ax.legend(
            title="Site", title_fontsize=8, fontsize=7,
            facecolor="#1a1a2e", edgecolor="#444444",
            labelcolor="white", framealpha=0.85,
            loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        )
        legend.get_title().set_color("#aaaaaa")
    else:
        for i, (x, y) in enumerate(coords[:, :2]):
            ax.scatter(x, y, color=_PALETTE[i % len(_PALETTE)], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5)

    # Overlay tool-type annotations on the dots
    if notes_map:
        import matplotlib.patheffects as pe
        for stem, (x, y) in zip(stems, coords[:, :2]):
            ann = notes_map.get(stem)
            if ann:
                ax.text(
                    x, y, ann,
                    ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white", zorder=6,
                    path_effects=[
                        pe.withStroke(linewidth=1.5, foreground="black"),
                    ],
                )

    ax.set_xlabel(f"PC1  ({var[0]:.1%} var)", color="#aaaaaa")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} var)", color="#aaaaaa")
    ax.set_title(f"Point-MAE embeddings — PCA  ({len(stems)} stones)",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    png = out / "pca.png"
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"PCA plot -> {png}")
    return png


# ---------------------------------------------------------------------------
# 3-D PCA  (interactive HTML)
# ---------------------------------------------------------------------------

def plot_pca_3d(stems, embeddings, out, site_map=None, notes_map=None):
    try:
        import plotly  # noqa: F401
    except ImportError:
        log.warning("plotly not installed; skipping 3D PCA.  pip install plotly")
        return None

    coords, var, _ = _scale_and_reduce(embeddings, n_components=3)
    axis_labels = [f"PC{i+1} ({var[i]:.1%})" for i in range(3)]
    title = f"Point-MAE embeddings — PCA 3D  ({len(stems)} stones)"

    fig = _plotly_3d(stems, coords, axis_labels, title, site_map, notes_map)
    html = out / "pca_3d.html"
    fig.write_html(str(html))
    log.info(f"3D PCA -> {html}")
    return html


# ---------------------------------------------------------------------------
# UMAP  (2-D PNG  +  3-D interactive HTML)
# ---------------------------------------------------------------------------

def plot_umap(stems, embeddings, out, site_map=None, notes_map=None, seed=42):
    try:
        import umap as umap_module
    except ImportError:
        log.warning("umap-learn not installed; skipping UMAP.  pip install umap-learn")
        return None, None

    E = np.stack(embeddings)
    E_scaled = StandardScaler().fit_transform(E)

    n_neighbors = min(15, len(stems) - 1)

    # ---- 2-D UMAP ----
    reducer2 = umap_module.UMAP(n_components=2, n_neighbors=n_neighbors,
                                random_state=seed)
    coords2 = reducer2.fit_transform(E_scaled)

    stem_sites, unique_sites, site_color = _site_colors(stems, site_map)

    fig2, ax = plt.subplots(figsize=(11, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    if site_map:
        for site in unique_sites:
            idxs = [i for i, s in enumerate(stem_sites) if s == site]
            ax.scatter(coords2[idxs, 0], coords2[idxs, 1],
                       color=site_color[site], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5, label=site)
        legend = ax.legend(
            title="Site", title_fontsize=8, fontsize=7,
            facecolor="#1a1a2e", edgecolor="#444444",
            labelcolor="white", framealpha=0.85,
            loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        )
        legend.get_title().set_color("#aaaaaa")
    else:
        for i, (x, y) in enumerate(coords2):
            ax.scatter(x, y, color=_PALETTE[i % len(_PALETTE)], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5)

    if notes_map:
        import matplotlib.patheffects as pe
        for stem, (x, y) in zip(stems, coords2):
            ann = notes_map.get(stem)
            if ann:
                ax.text(x, y, ann, ha="center", va="center",
                        fontsize=5, fontweight="bold", color="white", zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    ax.set_xlabel("UMAP-1", color="#aaaaaa")
    ax.set_ylabel("UMAP-2", color="#aaaaaa")
    ax.set_title(f"Point-MAE embeddings — UMAP  ({len(stems)} stones)",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    png = out / "umap.png"
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    log.info(f"UMAP 2D -> {png}")

    # ---- 3-D UMAP ----
    html = None
    try:
        import plotly  # noqa: F401
        reducer3 = umap_module.UMAP(n_components=3, n_neighbors=n_neighbors,
                                    random_state=seed)
        coords3 = reducer3.fit_transform(E_scaled)
        fig3 = _plotly_3d(stems, coords3,
                          ["UMAP-1", "UMAP-2", "UMAP-3"],
                          f"Point-MAE embeddings — UMAP 3D  ({len(stems)} stones)",
                          site_map, notes_map)
        html = out / "umap_3d.html"
        fig3.write_html(str(html))
        log.info(f"UMAP 3D -> {html}")
    except ImportError:
        log.warning("plotly not installed; skipping 3D UMAP HTML.")

    return png, html


# ---------------------------------------------------------------------------
# Scalar-coloured PCA (continuous colourmap)
# ---------------------------------------------------------------------------

def plot_scalar_pca(stems, embeddings, values, label, filename, out, notes_map=None):
    """2-D PCA scatter coloured by a continuous scalar (e.g. a bounding-box dimension)."""
    coords, var, _ = _scale_and_reduce(embeddings, n_components=10)

    vals = np.array(values, dtype=float)
    valid = np.isfinite(vals)
    if not valid.any():
        log.warning(f"No finite values for '{label}'; skipping {filename}")
        return None

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="plasma",
                    s=120, zorder=3, edgecolors="white", linewidths=0.5)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(label, color="#aaaaaa", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#555555")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa", fontsize=7)
    cbar.outline.set_edgecolor("#333333")

    if notes_map:
        import matplotlib.patheffects as pe
        for stem, (x, y) in zip(stems, coords[:, :2]):
            ann = notes_map.get(stem)
            if ann:
                ax.text(x, y, ann, ha="center", va="center",
                        fontsize=5, fontweight="bold", color="white", zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    ax.set_xlabel(f"PC1  ({var[0]:.1%} var)", color="#aaaaaa")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} var)", color="#aaaaaa")
    ax.set_title(f"Point-MAE embeddings — PCA by {label}  ({len(stems)} stones)",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    png = out / filename
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"PCA by {label} -> {png}")
    return png


def plot_scalar_pca_3d(stems, embeddings, values, label, filename, out, notes_map=None):
    """3-D PCA interactive HTML coloured by a continuous scalar."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        log.warning("plotly not installed; skipping 3D scalar PCA.  pip install plotly")
        return None

    coords, var, _ = _scale_and_reduce(embeddings, n_components=3)
    axis_labels = [f"PC{i+1} ({var[i]:.1%})" for i in range(3)]
    title = f"Point-MAE — PCA 3D by {label}  ({len(stems)} stones)"

    fig = _plotly_scalar_3d(stems, coords, values, axis_labels, title, notes_map)
    html = out / filename
    fig.write_html(str(html))
    log.info(f"3D scalar PCA ({label}) -> {html}")
    return html


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wrl_dir",       default=str(WRL_DIR))
    p.add_argument("--output_dir",    default=str(OUTPUT_DIR))
    p.add_argument("--n",             type=int, default=None, help="Number of files to process (default: all)")
    p.add_argument("--num_points",    type=int, default=2048)
    p.add_argument("--sampling_mode", default="edge_aware",
                   choices=["uniform", "curvature", "edge_aware"])
    p.add_argument("--checkpoint",    default=r"C:\Users\dshavit.WISMAIN\work\stones\pth\modelnet_8k.pth")
    p.add_argument("--device",        default=None)
    p.add_argument("--skip_clean",    action="store_true")
    p.add_argument("--decimate",      action="store_true")
    p.add_argument("--target_faces",  type=int, default=None)
    p.add_argument("--target_ratio",  type=float, default=0.05)
    p.add_argument("--decimate_method", default="qem", choices=["qem", "cluster"])
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--site_xlsx",     default=str(XLSX_DEFAULT),
                   help="Excel spreadsheet mapping WRL stems to archaeological sites")
    p.add_argument("--no_site_color", action="store_true",
                   help="Disable site-based coloring even if --site_xlsx is set")
    p.add_argument("--pca_only",      action="store_true",
                   help="Skip processing; reload saved embeddings and replot PCA")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cosine similarity matrix
# ---------------------------------------------------------------------------

def _sim_label_size(N):
    return max(2, min(7, int(180 / max(N, 1))))


def _sim_cell_size(N):
    return max(0.05, min(0.20, 12.0 / max(N, 1)))


def _draw_heatmap_axes(ax, sim, stems, stem_sites, site_color, vmin, vmax, fs):
    """Populate a single axes with the heatmap, coloured tick labels, no colorbar."""
    N = len(stems)
    # Mask diagonal so N/A cells don't anchor the colorscale
    sim_disp = sim.copy().astype(float)
    np.fill_diagonal(sim_disp, np.nan)
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="#444444")          # diagonal rendered as dark grey
    im = ax.imshow(sim_disp, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(stems, rotation=90, fontsize=fs, fontfamily="monospace")
    ax.set_yticklabels(stems, fontsize=fs, fontfamily="monospace")
    for tick, site in zip(ax.get_xticklabels(), stem_sites):
        tick.set_color(site_color[site])
    for tick, site in zip(ax.get_yticklabels(), stem_sites):
        tick.set_color(site_color[site])
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_facecolor("#0f0f1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    return im


def _sim_heatmap_simple(stems, sim, stem_sites, site_color, path, title, vmin, vmax):
    """Plain heatmap (no dendrogram)."""
    N    = len(stems)
    fs   = _sim_label_size(N)
    cell = _sim_cell_size(N)
    fig, ax = plt.subplots(figsize=(max(10, N * cell + 2), max(10, N * cell + 1)),
                           facecolor="#0f0f1a")
    im = _draw_heatmap_axes(ax, sim, stems, stem_sites, site_color, vmin, vmax, fs)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson r", color="#aaaaaa", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa", fontsize=7)
    cbar.outline.set_edgecolor("#333333")
    ax.set_title(title, color="white", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Similarity heatmap -> {path}")


def _sim_clustermap(stems, sim, stem_sites, site_color, linkage_Z, order,
                    path, title, vmin, vmax):
    """Heatmap with dendrograms on top/left and a site-colour strip on the left."""
    from scipy.cluster.hierarchy import dendrogram

    N      = len(stems)
    fs     = _sim_label_size(N)
    cell   = _sim_cell_size(N)
    heat   = N * cell
    dh     = max(1.2, heat * 0.12)   # dendrogram arm height
    strip  = max(0.15, heat * 0.025) # site colour strip width

    fig = plt.figure(facecolor="#0f0f1a",
                     figsize=(strip + dh + heat + 1.0, dh + heat + 0.6))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios  = [strip, dh, heat, 0.5],
        height_ratios = [dh, heat],
        hspace=0.02, wspace=0.02,
        left=0.01, right=0.99, top=0.97, bottom=0.03,
    )
    ax_dtop  = fig.add_subplot(gs[0, 2])   # top dendrogram
    ax_dleft = fig.add_subplot(gs[1, 1])   # left dendrogram
    ax_site  = fig.add_subplot(gs[1, 0])   # site colour strip
    ax_heat  = fig.add_subplot(gs[1, 2])   # heatmap
    ax_cbar  = fig.add_subplot(gs[1, 3])   # colorbar

    # Reorder
    stems_s = [stems[i]      for i in order]
    sites_s = [stem_sites[i] for i in order]
    sim_s   = sim[np.ix_(order, order)]

    # Dendrograms
    dc = "#666666"
    lkw = dict(no_labels=True, color_threshold=0,
               link_color_func=lambda _: dc)
    dendrogram(linkage_Z, ax=ax_dtop,  orientation="top",  **lkw)
    dendrogram(linkage_Z, ax=ax_dleft, orientation="left", **lkw)
    for ax in (ax_dtop, ax_dleft):
        ax.set_facecolor("#0f0f1a")
        ax.axis("off")

    # Site colour strip (one pixel-wide column per item, coloured by site)
    strip_rgb = np.array(
        [[matplotlib.colors.to_rgb(site_color[s]) for s in sites_s]]
    )                                        # [1, N, 3]
    strip_rgb = np.transpose(strip_rgb, (1, 0, 2))   # [N, 1, 3]
    ax_site.imshow(strip_rgb, aspect="auto", interpolation="nearest")
    ax_site.set_xticks([]); ax_site.set_yticks([])
    ax_site.set_facecolor("#0f0f1a")

    # Heatmap
    im = _draw_heatmap_axes(ax_heat, sim_s, stems_s, sites_s, site_color,
                            vmin, vmax, fs)

    # Colorbar
    plt.colorbar(im, cax=ax_cbar)
    ax_cbar.yaxis.tick_right()
    ax_cbar.yaxis.set_label_position("right")
    ax_cbar.set_ylabel("Pearson r", color="#aaaaaa", fontsize=8)
    plt.setp(ax_cbar.yaxis.get_ticklabels(), color="#aaaaaa", fontsize=7)

    fig.suptitle(title, color="white", fontsize=11)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Clustermap -> {path}")


def plot_similarity_matrix(stems, embeddings, out, site_map=None):
    E = np.stack(embeddings)
    sim = np.corrcoef(E).astype(np.float32)    # [N, N]
    np.fill_diagonal(sim, 1.0)

    np.save(out / "similarity_matrix.npy", sim)
    try:
        import pandas as pd
        pd.DataFrame(sim, index=stems, columns=stems).to_csv(
            out / "similarity_matrix.csv")
        log.info(f"Similarity CSV -> {out / 'similarity_matrix.csv'}")
    except Exception:
        pass

    stem_sites, _, site_color = _site_colors(stems, site_map)

    # Symmetric colormap range from the off-diagonal values
    mask = ~np.eye(len(stems), dtype=bool)
    vabs = float(np.abs(sim[mask]).max()) if mask.any() else 1.0
    vmin, vmax = -vabs, vabs

    _sim_heatmap_simple(
        stems, sim, stem_sites, site_color,
        out / "similarity_raw.png",
        f"Pearson correlation — raw order  ({len(stems)} objects)",
        vmin, vmax,
    )

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list

        Z     = linkage(E, method="ward", metric="euclidean")
        order = leaves_list(Z)

        _sim_clustermap(
            stems, sim, stem_sites, site_color, Z, order,
            out / "similarity_sorted.png",
            f"Pearson correlation — clustered  ({len(stems)} objects)",
            vmin, vmax,
        )
    except ImportError:
        log.warning("scipy not installed; skipping clustermap.  pip install scipy")


def _compute_dims_from_wrl(stems, wrl_dir: Path) -> dict:
    """
    Load each WRL, compute SVD-aligned bounding box extents, return {stem: [l,w,d]}.
    Fast path: no cleaning, no decimation, just vertices + SVD.
    """
    result = {}
    for stem in stems:
        # Try both .wrl and .WRL
        path = next(
            (wrl_dir / f"{stem}{ext}" for ext in (".wrl", ".WRL")
             if (wrl_dir / f"{stem}{ext}").exists()),
            None,
        )
        if path is None:
            log.warning(f"  {stem}: WRL file not found in {wrl_dir}, skipping dims")
            continue
        try:
            raw = load_mesh(path)
            v = raw["vertices"]
            centred = v - v.mean(axis=0)
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            proj = centred @ Vt.T
            extents = (proj.max(axis=0) - proj.min(axis=0)).tolist()
            result[stem] = extents
        except Exception as e:
            log.warning(f"  {stem}: could not compute dims ({e})")
    log.info(f"Computed dims for {len(result)}/{len(stems)} stems")
    return result


def _run_plots(stems, embeddings, out, site_map, notes_map,
               extents_map=None, seed=42):
    plot_pca(stems, embeddings, out, site_map=site_map, notes_map=notes_map)
    plot_pca_3d(stems, embeddings, out, site_map=site_map, notes_map=notes_map)
    plot_umap(stems, embeddings, out, site_map=site_map, notes_map=notes_map, seed=seed)
    plot_similarity_matrix(stems, embeddings, out, site_map=site_map)

    if extents_map:
        dims = [extents_map.get(s) for s in stems]
        if any(d is not None for d in dims):
            def col(i):
                return [d[i] if d is not None else float("nan") for d in dims]

            lengths = col(0)
            widths  = col(1)
            depths  = col(2)
            hw = [l/w if (w and w > 0) else float("nan")
                  for l, w in zip(lengths, widths)]
            wd = [w/d if (d and d > 0) else float("nan")
                  for w, d in zip(widths, depths)]

            scalars = [
                (lengths, "height (length)",    "pca_height"),
                (widths,  "width",              "pca_width"),
                (depths,  "depth",              "pca_depth"),
                (hw,      "height:width ratio", "pca_hw_ratio"),
                (wd,      "width:depth ratio",  "pca_wd_ratio"),
            ]
            for vals, label, stem_name in scalars:
                plot_scalar_pca(stems, embeddings, vals, label,
                                f"{stem_name}.png", out, notes_map)
                plot_scalar_pca_3d(stems, embeddings, vals, label,
                                   f"{stem_name}_3d.html", out, notes_map)


def _load_site_and_notes(args):
    """Load site_map + notes_map from xlsx if available."""
    if args.no_site_color:
        return None, None
    xlsx = Path(args.site_xlsx)
    if not xlsx.exists():
        log.warning(f"--site_xlsx not found: {xlsx}  (coloring by index)")
        return None, None
    try:
        return load_site_map(xlsx)
    except Exception as e:
        log.warning(f"Could not load site map: {e}")
        return None, None


def _run_folder(args) -> Path:
    return Path(args.output_dir)


def _run_incremental_plots(stems, embeddings, out, site_map, notes_map, seed=42):
    """
    Re-generate PCA, UMAP, and similarity matrix after each processed file.
    Overwrites the same output filenames every call so viewers see live updates.
    Scalar (height/width/depth) plots are skipped — those run at the end only.
    """
    import warnings
    n = len(stems)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plot_pca(stems, embeddings, out, site_map, notes_map)
            plot_pca_3d(stems, embeddings, out, site_map, notes_map)
            plot_umap(stems, embeddings, out, site_map, notes_map, seed=seed)
            plot_similarity_matrix(stems, embeddings, out, site_map)
        except Exception:
            log.debug(f"Incremental plot skipped (n={n}): {traceback.format_exc()[-200:]}")


def main():
    args = parse_args()
    out  = ensure_dir(_run_folder(args))

    # --pca_only: reload saved embeddings and replot without reprocessing
    if args.pca_only:
        emb_dir = out / "embeddings"
        files = sorted(emb_dir.glob("*.npy"))
        if not files:
            log.error(f"No saved embeddings found in {emb_dir}")
            return
        stems      = [f.stem for f in files]
        embeddings = [np.load(f) for f in files]
        log.info(f"Loaded {len(stems)} saved embeddings from {emb_dir}")

        import json
        dims_path = out / "dims.json"
        extents_map = {}
        if dims_path.exists():
            with open(dims_path) as fh:
                extents_map = json.load(fh)
            log.info(f"Loaded dims from {dims_path}")
        else:
            log.info("dims.json not found — computing SVD-aligned dims from WRL files...")
            extents_map = _compute_dims_from_wrl(stems, Path(args.wrl_dir))
            if extents_map:
                with open(dims_path, "w") as fh:
                    json.dump(extents_map, fh, indent=2)
                log.info(f"Saved dims -> {dims_path}")

        site_map, notes_map = _load_site_and_notes(args)
        _run_plots(stems, embeddings, out, site_map, notes_map,
                   extents_map=extents_map or None, seed=args.seed)
        return

    rng  = np.random.default_rng(args.seed)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta_writer = MetadataWriter(out / "metadata.csv")
    fail_writer = FailureWriter(out / "failed_meshes.csv")

    files = collect_mesh_files(args.wrl_dir, limit=args.n)
    log.info(f"Processing {len(files)} files from {args.wrl_dir}")
    log.info(f"Device: {device}  |  points: {args.num_points}  |  mode: {args.sampling_mode}"
             + ("  |  skip_clean" if args.skip_clean else "")
             + (f"  |  decimate to {args.target_ratio*100:.0f}%" if args.decimate else ""))

    embedder = build_embedder(checkpoint=args.checkpoint, device=device,
                              num_points=args.num_points)

    # Load site/notes map once before the loop so incremental plots can use it
    site_map, notes_map = _load_site_and_notes(args)

    results = []
    extents_map = {}
    for i, path in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {path.name}")
        try:
            r = process_one(path, embedder, args, out, meta_writer, fail_writer, rng)
            if r is not None:
                results.append(r)
                extents_map[r[0]] = r[2].tolist()
                if len(results) >= 2:
                    stems_so_far = [x[0] for x in results]
                    embs_so_far  = [x[1] for x in results]
                    _run_incremental_plots(
                        stems_so_far, embs_so_far, out,
                        site_map, notes_map,
                        seed=args.seed,
                    )
        except Exception:
            log.error(traceback.format_exc()[-300:])
            fail_writer.write(path.name, "pipeline", "unhandled exception")

    # Outputs summary
    log.info("")
    log.info("Outputs")
    log.info(f"  embeddings/     {out / 'embeddings'}")
    log.info(f"  pointclouds/    {out / 'pointclouds'}")
    log.info(f"  metadata.csv    {out / 'metadata.csv'}")
    log.info(f"  failed_meshes   {out / 'failed_meshes.csv'}")

    if not results:
        log.error("No embeddings produced.")
        return

    # Consolidate
    stems      = [r[0] for r in results]
    embeddings = [r[1] for r in results]
    all_E = np.stack(embeddings)
    np.save(out / "all_embeddings.npy", all_E)
    import pandas as pd
    pd.DataFrame(all_E, index=stems).to_csv(out / "all_embeddings.csv", index_label="stem")
    log.info(f"  all_embeddings  {out / 'all_embeddings.npy'}  shape={all_E.shape}")

    # Save SVD-aligned bounding box dims for --pca_only reruns
    import json
    with open(out / "dims.json", "w") as fh:
        json.dump(extents_map, fh, indent=2)

    if len(results) >= 2:
        _run_plots(stems, embeddings, out, site_map, notes_map,
                   extents_map=extents_map, seed=args.seed)
    else:
        log.warning("Need at least 2 embeddings for plots.")


if __name__ == "__main__":
    main()
