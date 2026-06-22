"""
DINOv2 embedding pipeline for stone tool WRL meshes — 7-channel image input.

Network input: a 7-channel image [H, W, 7] per stone, composed of:
  ch 0-2  top-surface normals    (Nx, Ny, Nz), encoded as (n+1)/2 ∈ [0,1]
  ch 3-5  bottom-surface normals (Nx, Ny, Nz), encoded as (n+1)/2 ∈ [0,1]
  ch 6    top-surface Z-depth,   encoded as (Z - Z_base) / global_max_Z_range

All normals are guaranteed to point away from the mesh centroid.

Scale convention
----------------
A single pixel-per-unit scale is derived from the maximum X extent and maximum
Y extent observed across ALL objects (tracked separately, not combined into one
number).  The canvas width equals the largest X extent across all objects; the
canvas height equals the largest Y extent.  No per-object scaling is applied —
each stone's geometry is placed at the common scale so relative sizes and aspect
ratios are preserved.  The canvas is zero-padded to a square before being fed
to the DINOv2 transformer.

Depth convention
----------------
Depth at each (X, Y) pixel is the Z-coordinate of the top (highest-Z) visible
surface at that point.  It is expressed relative to the object's own Z minimum
(so 0 = base of the object) and normalised by the global maximum Z-range across
all objects.  This keeps relative depths comparable across stones.

Usage
-----
  python scripts/demo_dino7ch_embed.py
  python scripts/demo_dino7ch_embed.py --skip_clean
  python scripts/demo_dino7ch_embed.py --decimate --target_ratio 0.05
  python scripts/demo_dino7ch_embed.py --reload_scale outputs/global_scale.json
  python scripts/demo_dino7ch_embed.py --pca_only    # replot cached embeddings
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_cleaning import clean_mesh, compute_face_normals
from src.mesh_io import load_mesh
from src.utils import collect_mesh_files, ensure_dir, get_logger
from scripts.demo_embed_pca import (
    load_site_map,
    plot_similarity_matrix,
    _PALETTE,
    _scale_and_reduce,
    _site_colors,
    _plotly_3d,
)

log = get_logger("dino7ch_embed")

IMAGE_SIZE = 224
MARGIN     = 0.05        # fraction of each canvas axis kept as black border
DINO_MODEL = "dinov2_vits14"
N_CH       = 7
PATCH_SIZE = 14

WRL_DIR      = ROOT / "wrl"
OUTPUT_BASE  = ROOT / "outputs" / "dino_7ch_v2"   # timestamped subfolder added at runtime
SCALE_DIR    = ROOT / "outputs"                    # global_scale.json always lives here
XLSX_DEFAULT = WRL_DIR / "Handaxes 2026 list with sites.xlsx"


# ---------------------------------------------------------------------------
# Pass 1 — global scale and depth range
# ---------------------------------------------------------------------------

def _raw_extents(path: Path) -> tuple[float, float, float]:
    """
    Load raw vertices (no cleaning) and return (X_extent, Y_extent, Z_extent).
    Returns (0, 0, 0) on failure.
    """
    try:
        v = load_mesh(path)["vertices"]
        ext = v.max(axis=0) - v.min(axis=0)
        return float(ext[0]), float(ext[1]), float(ext[2])
    except Exception:
        return 0.0, 0.0, 0.0


def find_global_params(
    files: list[Path],
    img_size: int,
    margin: float,
) -> tuple[float, int, int, float]:
    """
    Scan all meshes once (vertices only, no cleaning) to determine:

      scale       — pixels per world-unit, chosen so the object with the
                    largest extent in X or Y fills `img_size*(1-2*margin)` px.
      canvas_W    — image width in pixels  = ceil(max_X_all * scale + 2*margin_px)
      canvas_H    — image height in pixels = ceil(max_Y_all * scale + 2*margin_px)
      max_z_range — maximum Z-extent (top - bottom) across all objects;
                    used to normalise the depth channel globally.

    max_X and max_Y are tracked independently so the canvas reflects the true
    largest width and height across the dataset.
    """
    log.info(f"Pass 1: scanning {len(files)} meshes for global scale & depth …")
    max_x = max_y = max_z = 0.0
    for p in files:
        ex, ey, ez = _raw_extents(p)
        max_x = max(max_x, ex)
        max_y = max(max_y, ey)
        max_z = max(max_z, ez)

    max_x = max(max_x, 1e-10)
    max_y = max(max_y, 1e-10)
    max_z = max(max_z, 1e-10)

    inner    = img_size * (1.0 - 2.0 * margin)
    margin_px = img_size * margin

    # Single pixel scale so the largest axis (X or Y) fills `inner` pixels.
    scale    = inner / max(max_x, max_y)
    canvas_W = max(1, round(max_x * scale + 2.0 * margin_px))
    canvas_H = max(1, round(max_y * scale + 2.0 * margin_px))

    log.info(
        f"  max X={max_x:.4f}  max Y={max_y:.4f}  max Z={max_z:.4f}  "
        f"scale={scale:.3f} px/unit  canvas={canvas_W}x{canvas_H}  "
        f"max_z_range={max_z:.4f}"
    )
    return scale, canvas_W, canvas_H, max_z


SCALE_FILE = SCALE_DIR / "global_scale.json"


def save_global_params(scale: float, canvas_W: int, canvas_H: int,
                       max_z_range: float, path: Path = SCALE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"scale": scale, "canvas_W": canvas_W,
                   "canvas_H": canvas_H, "max_z_range": max_z_range}, fh, indent=2)
    log.info(f"Global scale saved → {path}")


def load_global_params(path: Path = SCALE_FILE) -> tuple[float, int, int, float]:
    with open(path) as fh:
        d = json.load(fh)
    scale, canvas_W, canvas_H, max_z = (
        float(d["scale"]), int(d["canvas_W"]),
        int(d["canvas_H"]), float(d["max_z_range"]),
    )
    log.info(
        f"Global scale loaded from {path}\n"
        f"  scale={scale:.3f} px/unit  canvas={canvas_W}x{canvas_H}"
        f"  max_z_range={max_z:.4f}"
    )
    return scale, canvas_W, canvas_H, max_z


# ---------------------------------------------------------------------------
# Renderer — 7-channel image
# ---------------------------------------------------------------------------

def _poly_render(
    polys: np.ndarray,
    colors: np.ndarray,
    canvas_H: int,
    canvas_W: int,
) -> np.ndarray:
    """
    Rasterise triangles onto a black canvas using matplotlib PolyCollection.

    polys  : [F, 3, 2]  triangle vertices in pixel coordinates (x, y)
    colors : [F, 3]     RGB colours in [0, 1]

    Returns [canvas_H, canvas_W, 3] uint8.
    """
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(canvas_W / dpi, canvas_H / dpi), dpi=dpi
    )
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, canvas_W)
    ax.set_ylim(0, canvas_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor((0, 0, 0))
    fig.patch.set_facecolor((0, 0, 0))

    col = PolyCollection(polys, facecolors=colors, edgecolors="none", antialiased=False)
    ax.add_collection(col)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    if h != canvas_H or w != canvas_W:
        buf = np.array(
            Image.fromarray(buf).resize((canvas_W, canvas_H), Image.BILINEAR)
        )
    return buf


def render_7ch(
    vertices: np.ndarray,
    faces: np.ndarray,
    scale: float,
    canvas_H: int,
    canvas_W: int,
    global_max_z_range: float,
) -> np.ndarray:
    """
    Render a 7-channel image for one mesh.

    All normals are flipped so they point away from the mesh centroid before
    splitting into top / bottom views.

    Returns [canvas_H, canvas_W, 7] float32 in [0, 1]:
      ch 0-2  top-surface normal  (Nx, Ny, Nz) encoded as (n+1)/2
      ch 3-5  bottom-surface normal
      ch 6    top-surface Z-depth, (Z - Z_min) / global_max_z_range
    """
    # ---- Step 1: face normals, oriented away from centroid ----
    centroid = vertices.mean(axis=0)

    normals = compute_face_normals(vertices, faces).copy()  # [F, 3]

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_cents  = (v0 + v1 + v2) / 3.0         # [F, 3]
    to_face     = face_cents - centroid          # vector from centroid to face centre
    flip_mask   = np.einsum("fi,fi->f", normals, to_face) < 0
    normals[flip_mask] = -normals[flip_mask]     # flip inward-pointing normals

    # ---- Step 2: project to pixel space, centred in canvas ----
    xy_min = vertices[:, :2].min(axis=0)
    xy_max = vertices[:, :2].max(axis=0)
    ctr    = (xy_min + xy_max) * 0.5

    px = (vertices[:, 0] - ctr[0]) * scale + canvas_W / 2.0
    py = (vertices[:, 1] - ctr[1]) * scale + canvas_H / 2.0
    pz = vertices[:, 2]

    v2d   = np.stack([px, py], axis=1)  # [V, 2]
    tris  = v2d[faces]                   # [F, 3, 2]
    avg_z = pz[faces].mean(axis=1)      # [F] mean Z for painter ordering

    # ---- Step 3: split top / bottom faces ----
    # Top:    normal Nz >= 0  (points upward / away from centroid toward camera)
    # Bottom: normal Nz <= 0  (points downward / away from centroid from below)
    top_mask = normals[:, 2] >= 0
    bot_mask = normals[:, 2] <= 0

    top_idx = np.where(top_mask)[0]
    bot_idx = np.where(bot_mask)[0]

    # Painter's order: back-to-front so foreground faces overwrite background
    if len(top_idx):
        top_idx = top_idx[np.argsort(avg_z[top_idx])]      # lowest Z first
    if len(bot_idx):
        bot_idx = bot_idx[np.argsort(-avg_z[bot_idx])]     # highest Z first (bottom view)

    enc = lambda n: np.clip((n + 1.0) * 0.5, 0.0, 1.0)    # normal → [0,1]

    # ---- Step 4: render top normals ----
    if len(top_idx):
        rgb = _poly_render(tris[top_idx], enc(normals[top_idx]), canvas_H, canvas_W)
        top_f = rgb.astype(np.float32) / 255.0
    else:
        top_f = np.zeros((canvas_H, canvas_W, 3), np.float32)

    # ---- Step 5: render bottom normals ----
    if len(bot_idx):
        rgb = _poly_render(tris[bot_idx], enc(normals[bot_idx]), canvas_H, canvas_W)
        bot_f = rgb.astype(np.float32) / 255.0
    else:
        bot_f = np.zeros((canvas_H, canvas_W, 3), np.float32)

    # ---- Step 6: depth channel (top-surface Z, globally normalised) ----
    if len(top_idx):
        z_base = float(pz.min())
        z_vals = avg_z[top_idx] - z_base          # shift so object base = 0
        if global_max_z_range > 1e-10:
            d = np.clip(z_vals / global_max_z_range, 0.0, 1.0)
        else:
            d = np.full(len(top_idx), 0.5, dtype=np.float32)
        rgb = _poly_render(
            tris[top_idx],
            np.stack([d, d, d], axis=1),
            canvas_H, canvas_W,
        )
        dep_f = rgb[:, :, :1].astype(np.float32) / 255.0   # keep only 1 ch
    else:
        dep_f = np.zeros((canvas_H, canvas_W, 1), np.float32)

    return np.concatenate([top_f, bot_f, dep_f], axis=2)   # [H, W, 7]


def save_renders(img7: np.ndarray, stem: str, rd: Path) -> None:
    """
    Save the 7-channel image as three PNGs inside `rd/`:
      <stem>_top.png   — RGB  top-surface normals
      <stem>_bot.png   — RGB  bottom-surface normals
      <stem>_depth.png — L (grayscale) top-surface depth
    """
    rd.mkdir(parents=True, exist_ok=True)

    def to_uint8(arr: np.ndarray) -> np.ndarray:
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    Image.fromarray(to_uint8(img7[:, :, :3])).save(rd / f"{stem}_top.png")
    Image.fromarray(to_uint8(img7[:, :, 3:6])).save(rd / f"{stem}_bot.png")
    Image.fromarray(to_uint8(img7[:, :, 6]), mode="L").save(rd / f"{stem}_depth.png")


# ---------------------------------------------------------------------------
# 7-channel DINOv2 embedder
# ---------------------------------------------------------------------------

def build_dino7ch(
    model_name: str = DINO_MODEL,
    checkpoint: str | None = None,
    device: torch.device | None = None,
    n_channels: int = N_CH,
) -> nn.Module:
    """
    Load DINOv2 and widen the patch-embedding Conv2d from 3 → n_channels.

    Weight initialisation: average the pretrained RGB weights across the
    channel axis (one prototype weight), replicate n_channels times, and scale
    by 3/n_channels to maintain the expected pre-activation magnitude.
    """
    log.info(f"Loading DINOv2 {model_name} …")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)

    old = model.patch_embed.proj                              # Conv2d(3, D, P, P)
    D, P, has_bias = old.out_channels, old.kernel_size[0], old.bias is not None

    new = nn.Conv2d(n_channels, D, kernel_size=P, stride=P, bias=has_bias)
    with torch.no_grad():
        w_mean = old.weight.data.mean(dim=1, keepdim=True)   # [D, 1, P, P]
        new.weight.data = w_mean.repeat(1, n_channels, 1, 1) * (3.0 / n_channels)
        if has_bias:
            new.bias.data = old.bias.data.clone()
    model.patch_embed.proj = new
    log.info(f"  patch_embed widened: 3 → {n_channels} channels  (embed_dim={D})")

    if checkpoint and Path(checkpoint).exists():
        log.info(f"  loading checkpoint: {checkpoint}")
        ckpt  = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        miss, unex = model.load_state_dict(state, strict=False)
        log.info(
            f"  {len(state)-len(unex)}/{len(state)} keys matched "
            f"({len(miss)} missing, {len(unex)} unexpected)"
        )
    elif checkpoint:
        log.warning(f"  checkpoint not found: {checkpoint}  (using pretrained only)")

    model.eval()
    return model.to(device) if device is not None else model


@torch.no_grad()
def embed_7ch(
    model: nn.Module,
    img7: np.ndarray,
    device: torch.device,
    img_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Embed a 7-channel float32 [H, W, 7] image via the widened DINOv2.

    1. Zero-pad to a square canvas (largest of H, W).
    2. Resize to img_size × img_size if the square is not already that size.
    3. Normalise each channel from [0, 1] → [-1, 1].
    4. Forward pass; return the CLS-token vector [D].
    """
    H, W, _ = img7.shape
    sq = max(H, W)

    # Pad to square (centre the object)
    if H != W:
        padded = np.zeros((sq, sq, 7), dtype=np.float32)
        ph = (sq - H) // 2
        pw = (sq - W) // 2
        padded[ph:ph + H, pw:pw + W] = img7
        img7 = padded

    # Resize to model's expected input size
    if sq != img_size:
        channels = [
            np.array(
                Image.fromarray(
                    (np.clip(img7[:, :, c], 0, 1) * 255).astype(np.uint8)
                ).resize((img_size, img_size), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            for c in range(7)
        ]
        img7 = np.stack(channels, axis=2)

    # [1, 7, H, W], normalised to [-1, 1]
    t = torch.from_numpy(img7.transpose(2, 0, 1)).unsqueeze(0).to(device)
    t = t * 2.0 - 1.0
    return model(t).cpu().numpy()[0].astype(np.float32)     # [D]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_2d(stems, coords, site_map, notes_map, title, xlabel, ylabel, out_path):
    stem_sites, unique_sites, site_color = _site_colors(stems, site_map)
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    if site_map:
        for site in unique_sites:
            idxs = [i for i, s in enumerate(stem_sites) if s == site]
            ax.scatter(coords[idxs, 0], coords[idxs, 1],
                       color=site_color[site], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5, label=site)
        leg = ax.legend(
            title="Site", title_fontsize=8, fontsize=7,
            facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white",
            framealpha=0.85, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        )
        leg.get_title().set_color("#aaaaaa")
    else:
        for i, (x, y) in enumerate(coords[:, :2]):
            ax.scatter(x, y, color=_PALETTE[i % len(_PALETTE)], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5)
    if notes_map:
        import matplotlib.patheffects as pe
        for stem, (x, y) in zip(stems, coords[:, :2]):
            ann = notes_map.get(stem)
            if ann:
                ax.text(x, y, ann, ha="center", va="center",
                        fontsize=5, fontweight="bold", color="white", zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
    ax.set_xlabel(xlabel, color="#aaaaaa")
    ax.set_ylabel(ylabel, color="#aaaaaa")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  -> {out_path}")


def plot_pca(stems, embeddings, out, site_map=None, notes_map=None):
    coords, var, _ = _scale_and_reduce(embeddings, n_components=10)
    _plot_2d(stems, coords, site_map, notes_map,
             f"DINOv2-7ch embeddings — PCA  ({len(stems)} stones)",
             f"PC1  ({var[0]:.1%} var)", f"PC2  ({var[1]:.1%} var)",
             out / "pca.png")


def plot_pca_3d(stems, embeddings, out, site_map=None, notes_map=None):
    try:
        import plotly  # noqa: F401
    except ImportError:
        return
    coords, var, _ = _scale_and_reduce(embeddings, n_components=3)
    axis_labels = [f"PC{i+1} ({var[i]:.1%})" for i in range(3)]
    fig = _plotly_3d(stems, coords, axis_labels,
                     f"DINOv2-7ch embeddings — PCA 3D  ({len(stems)} stones)",
                     site_map, notes_map)
    html = out / "pca_3d.html"
    fig.write_html(str(html))
    log.info(f"  -> {html}")


def plot_umap(stems, embeddings, out, site_map=None, notes_map=None, seed=42):
    try:
        import umap as _umap
    except ImportError:
        log.warning("umap-learn not installed; skipping UMAP.  pip install umap-learn")
        return
    from sklearn.preprocessing import StandardScaler
    E    = StandardScaler().fit_transform(np.stack(embeddings))
    n_nb = min(15, len(stems) - 1)
    c2   = _umap.UMAP(n_components=2, n_neighbors=n_nb, random_state=seed).fit_transform(E)
    _plot_2d(stems, c2, site_map, notes_map,
             f"DINOv2-7ch embeddings — UMAP  ({len(stems)} stones)",
             "UMAP-1", "UMAP-2", out / "umap.png")
    try:
        import plotly  # noqa: F401
        c3 = _umap.UMAP(n_components=3, n_neighbors=n_nb, random_state=seed).fit_transform(E)
        fig3 = _plotly_3d(stems, c3, ["UMAP-1", "UMAP-2", "UMAP-3"],
                          f"DINOv2-7ch embeddings — UMAP 3D  ({len(stems)} stones)",
                          site_map, notes_map)
        html = out / "umap_3d.html"
        fig3.write_html(str(html))
        log.info(f"  -> {html}")
    except ImportError:
        pass


def _run_plots(stems, embeddings, out, site_map, notes_map, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plot_pca(stems, embeddings, out, site_map, notes_map)
            plot_pca_3d(stems, embeddings, out, site_map, notes_map)
            plot_umap(stems, embeddings, out, site_map, notes_map, seed=seed)
            plot_similarity_matrix(stems, embeddings, out, site_map)
        except Exception:
            log.debug(f"Plot skipped (n={len(stems)}): {traceback.format_exc()[-200:]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wrl_dir",         default=str(WRL_DIR))
    p.add_argument("--output_dir",      default=str(OUTPUT_BASE),
                   help="Base output folder; a timestamp subfolder is created inside")
    p.add_argument("--site_xlsx",       default=str(XLSX_DEFAULT),
                   help="Excel file mapping WRL stems to sites (optional)")
    p.add_argument("--no_site_color",   action="store_true")
    p.add_argument("--dino_model",      default=DINO_MODEL,
                   help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14")
    p.add_argument("--image_size",      type=int, default=IMAGE_SIZE,
                   help="Square canvas size fed to DINOv2 (multiple of 14)")
    p.add_argument("--device",          default=None)
    p.add_argument("--reload_scale",    default=None, metavar="JSON",
                   help="Path to a saved global_scale.json; skip pass-1 scan")
    p.add_argument("--limit",           type=int, default=None,
                   help="Process only the first N WRL files")
    p.add_argument("--skip_clean",      action="store_true",
                   help="Skip mesh cleaning (use raw geometry)")
    p.add_argument("--decimate",        action="store_true",
                   help="Decimate mesh before rendering")
    p.add_argument("--target_faces",    type=int, default=None)
    p.add_argument("--target_ratio",    type=float, default=0.05)
    p.add_argument("--decimate_method", default="qem", choices=["qem", "cluster"])
    p.add_argument("--pca_only",        action="store_true",
                   help="Skip rendering; reload cached embeddings and replot")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def _load_site_and_notes(args):
    if args.no_site_color:
        return None, None
    xlsx = Path(args.site_xlsx)
    if not xlsx.exists():
        log.warning(f"site_xlsx not found: {xlsx}  (no site colouring)")
        return None, None
    try:
        return load_site_map(xlsx)
    except Exception as e:
        log.warning(f"Could not load site map: {e}")
        return None, None


def _process_mesh(path: Path, args) -> tuple[np.ndarray, np.ndarray] | None:
    """Load → clean → decimate. Returns (vertices, faces) or None on failure."""
    raw = load_mesh(path)
    v, f = raw["vertices"], raw["faces"]
    log.info(f"  loaded  {len(v):>8,}V  {len(f):>8,}F")

    if not args.skip_clean:
        v, f, meta = clean_mesh(v, f, keep_largest_component=True)
        log.info(f"  clean   {meta['num_clean_vertices']:>8,}V  {meta['num_clean_faces']:>8,}F")

    if len(f) == 0:
        log.warning("  No faces — skipped")
        return None

    if args.decimate:
        from src.decimation import decimate
        from src.mesh_cleaning import orient_normals_outward
        tf = args.target_faces or max(500, int(len(f) * args.target_ratio))
        try:
            v, f = decimate(v, f, tf, method=args.decimate_method)
            f    = orient_normals_outward(v, f)
            log.info(f"  decimate {len(v):>8,}V  {len(f):>8,}F  (target {tf})")
        except Exception as e:
            log.warning(f"  Decimate failed ({e}), continuing with full mesh")

    return v, f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = ensure_dir(Path(args.output_dir) / stamp)
    emb_dir = out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    site_map, notes_map = _load_site_and_notes(args)

    # --pca_only: reload cached embeddings and replot
    if args.pca_only:
        files = sorted(emb_dir.glob("*.npy"))
        if not files:
            log.error(f"No embeddings found in {emb_dir}  (run without --pca_only first)")
            return
        stems      = [f.stem for f in files]
        embeddings = [np.load(f) for f in files]
        log.info(f"Loaded {len(stems)} cached embeddings  dim={embeddings[0].shape[0]}")
        _run_plots(stems, embeddings, out, site_map, notes_map, args.seed)
        return

    files = collect_mesh_files(args.wrl_dir, limit=args.limit)
    if not files:
        log.error(f"No WRL files found in {args.wrl_dir}")
        return
    log.info(f"Found {len(files)} WRL files")

    # Pass 1 — global scale and depth range (or reload from file)
    if args.reload_scale:
        scale, canvas_W, canvas_H, max_z_range = load_global_params(
            Path(args.reload_scale)
        )
    else:
        scale, canvas_W, canvas_H, max_z_range = find_global_params(
            files, args.image_size, MARGIN
        )
        save_global_params(scale, canvas_W, canvas_H, max_z_range)

    # Load 7-channel DINOv2
    model = build_dino7ch(args.dino_model, device=device)
    D     = model.embed_dim
    log.info(
        f"DINOv2-7ch {args.dino_model}  embed_dim={D}  device={device}\n"
        f"  canvas: {canvas_W}W × {canvas_H}H  →  {args.image_size}² (padded+resized)\n"
        f"  skip_clean={args.skip_clean}  decimate={args.decimate}"
    )

    renders_dir = out / "renders"

    stems:      list[str]        = []
    embeddings: list[np.ndarray] = []

    # Pass 2 — render + embed
    for i, path in enumerate(files, 1):
        stem     = path.stem
        emb_path = emb_dir / f"{stem}.npy"

        if emb_path.exists():
            emb = np.load(emb_path)
            stems.append(stem)
            embeddings.append(emb)
            log.info(f"[{i}/{len(files)}] {stem}  (cached)")
        else:
            log.info(f"[{i}/{len(files)}] {stem}")
            try:
                result = _process_mesh(path, args)
                if result is None:
                    continue
                v, f = result

                img7 = render_7ch(v, f, scale, canvas_H, canvas_W, max_z_range)

                save_renders(img7, stem, renders_dir)

                emb = embed_7ch(model, img7, device, args.image_size)
                np.save(emb_path, emb)
                stems.append(stem)
                embeddings.append(emb)
                log.info(f"  embed   dim={emb.shape[0]}")

            except Exception:
                log.error(f"  Failed:\n{traceback.format_exc()[-400:]}")
                continue

        if len(embeddings) >= 2:
            _run_plots(stems, embeddings, out, site_map, notes_map, args.seed)

    if not stems:
        log.error("No embeddings produced.")
        return

    all_E = np.stack(embeddings)
    np.save(out / "all_embeddings.npy", all_E)
    log.info(f"Done. {len(stems)} embeddings  shape={all_E.shape}")
    log.info(f"Output: {out}")

    if len(stems) >= 2:
        _run_plots(stems, embeddings, out, site_map, notes_map, args.seed)


if __name__ == "__main__":
    main()
