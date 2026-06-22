"""
DINOv2 embedding pipeline for stone tool WRL meshes.

Renders each WRL into three orthographic views:
  top_normal    — outward surface normals viewed from above (+Z)  → RGB
  bottom_normal — outward surface normals viewed from below (-Z)  → RGB
  depth         — normalised Z-depth, top-down                    → greyscale

Normal encoding: pixel = (n + 1) / 2  (maps [-1, 1] → [0, 1] per channel).
Depth encoding:  pixel = (Z - Z_min) / (Z_max - Z_min), greyscale.

The three views are packed into a single 7-channel image [H, W, 7] and passed
through DINOv2-ViT-S/14 with the patch-embedding conv widened from 3 → 7
channels.  The CLS token produces a 384-d embedding per stone.

Scale: a quick first pass loads all mesh vertex clouds (no cleaning) and finds
the maximum XY bounding-box extent across all objects.  That single scale is
applied to every render so relative sizes and aspect ratios are preserved.

Usage
-----
  python scripts/demo_dino_embed_7ch.py
  python scripts/demo_dino_embed_7ch.py --save_renders
  python scripts/demo_dino_embed_7ch.py --skip_clean
  python scripts/demo_dino_embed_7ch.py --decimate --target_ratio 0.05
  python scripts/demo_dino_embed_7ch.py --pca_only    # replot cached embeddings
"""
from __future__ import annotations

import argparse
import sys
import traceback
import warnings
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

log = get_logger("dino_embed_7ch")

IMAGE_SIZE = 224
MARGIN     = 0.05        # black border fraction
DINO_MODEL = "dinov2_vits14"
N_CH       = 7
PATCH_SIZE = 14

WRL_DIR      = ROOT / "wrl"
OUTPUT_DIR   = ROOT / "outputs" / "dino_7ch"
XLSX_DEFAULT = WRL_DIR / "Handaxes 2026 list with sites.xlsx"


# ---------------------------------------------------------------------------
# Global scale (pass 1: quick vertex-only scan)
# ---------------------------------------------------------------------------

def _xy_extent(path: Path) -> float:
    """Load vertices only; return max XY bounding-box dimension (no cleaning)."""
    try:
        raw = load_mesh(path)
        v   = raw["vertices"]
        return float((v[:, :2].max(axis=0) - v[:, :2].min(axis=0)).max())
    except Exception:
        return 0.0


def find_global_scale(files: list[Path], img_size: int, margin: float) -> float:
    """
    Pass 1: scan all meshes for their raw XY extents.
    Returns pixels-per-world-unit so the largest object fills the canvas.
    Preserves relative sizes and aspect ratios across the whole dataset.
    """
    log.info(f"Pass 1: scanning {len(files)} meshes for global scale …")
    max_ext = max((_xy_extent(p) for p in files), default=1.0)
    if max_ext < 1e-10:
        max_ext = 1.0
    scale = img_size * (1.0 - 2.0 * margin) / max_ext
    log.info(f"  max XY extent = {max_ext:.4f}  →  scale = {scale:.2f} px/unit")
    return scale


def _center_offset(vertices: np.ndarray, scale: float, img_size: int) -> tuple[float, float]:
    """Translate so the object is centred in the canvas at the given scale."""
    ctr = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) * 0.5
    cx  = img_size / 2.0 - ctr[0] * scale
    cy  = img_size / 2.0 - ctr[1] * scale
    return cx, cy


# ---------------------------------------------------------------------------
# Renderer (matplotlib PolyCollection + painter's algorithm)
# ---------------------------------------------------------------------------

def _visible_sorted(faces: np.ndarray, vz: np.ndarray,
                    face_normals: np.ndarray, from_top: bool) -> np.ndarray:
    """
    Back-face cull (keep faces whose outward Z-normal faces the camera)
    then sort survivors back-to-front (painter's algorithm).
    Returns an index array into `faces`.
    """
    vis = face_normals[:, 2] >= 0 if from_top else face_normals[:, 2] <= 0
    idx = np.where(vis)[0]
    if len(idx) == 0:
        return idx
    avg_z = vz[faces[idx]].mean(axis=1)
    return idx[np.argsort(avg_z if from_top else -avg_z)]


def _poly_render(polys: np.ndarray, colors: np.ndarray, img_size: int) -> np.ndarray:
    """
    Rasterise triangles onto a black canvas with matplotlib PolyCollection.
    polys  : [F, 3, 2]  pixel coordinates
    colors : [F, 3]     RGB in [0, 1]
    Returns [H, W, 3] uint8.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
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

    if h != img_size or w != img_size:
        buf = np.array(Image.fromarray(buf).resize((img_size, img_size), Image.BILINEAR))
    return buf


def render_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    scale: float,
    img_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render three views of a mesh at the given scale (pixels per world unit).

    Coordinate convention
    ---------------------
    Top view   : orthographic XY projection, camera at +Z.
                 Visible faces: outward normal Nz ≥ 0.
    Bottom view: same XY projection, camera at −Z.
                 Visible faces: outward normal Nz ≤ 0.
    Depth      : Z-value of the frontmost top-view surface, normalised to [0,1],
                 rendered as greyscale.

    X→column, Y→row, Y increasing upward in both world and image.

    Returns (top_normal, bottom_normal, depth) as [H, W, 3] uint8 arrays.
    Normals encoded: pixel = (n + 1) / 2  per channel.
    """
    face_normals = compute_face_normals(vertices, faces)
    cx, cy = _center_offset(vertices, scale, img_size)

    vx = vertices[:, 0] * scale + cx
    vy = vertices[:, 1] * scale + cy
    vz = vertices[:, 2]

    v_px = np.stack([vx, vy], axis=1)  # [V, 2]
    tris = v_px[faces]                  # [F, 3, 2]

    # ---- top normal ----
    idx_top = _visible_sorted(faces, vz, face_normals, from_top=True)
    if len(idx_top):
        top_normal = _poly_render(
            tris[idx_top],
            np.clip((face_normals[idx_top] + 1.0) * 0.5, 0.0, 1.0),
            img_size,
        )
    else:
        top_normal = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # ---- bottom normal ----
    idx_bot = _visible_sorted(faces, vz, face_normals, from_top=False)
    if len(idx_bot):
        bottom_normal = _poly_render(
            tris[idx_bot],
            np.clip((face_normals[idx_bot] + 1.0) * 0.5, 0.0, 1.0),
            img_size,
        )
    else:
        bottom_normal = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # ---- depth (top view) ----
    if len(idx_top):
        avg_z = vz[faces[idx_top]].mean(axis=1)
        z_min, z_max = avg_z.min(), avg_z.max()
        d = (avg_z - z_min) / (z_max - z_min) if z_max > z_min else np.full(len(idx_top), 0.5)
        depth = _poly_render(tris[idx_top], np.stack([d, d, d], axis=1), img_size)
    else:
        depth = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    return top_normal, bottom_normal, depth


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
    Load DINOv2 and widen the patch-embedding conv from 3 → n_channels.

    Initialisation: average the pretrained RGB weights across the channel axis
    (giving one channel prototype), replicate it n_channels times, and scale by
    3 / n_channels to maintain the expected pre-activation magnitude.
    """
    log.info(f"Loading DINOv2 {model_name} …")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)

    old = model.patch_embed.proj                                 # Conv2d(3, D, P, P)
    D   = old.out_channels
    P   = old.kernel_size[0]
    has_bias = old.bias is not None

    new = nn.Conv2d(n_channels, D, kernel_size=P, stride=P, bias=has_bias)
    with torch.no_grad():
        w_mean = old.weight.data.mean(dim=1, keepdim=True)      # [D, 1, P, P]
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
        log.info(f"  {len(state)-len(unex)}/{len(state)} keys matched "
                 f"({len(miss)} missing, {len(unex)} unexpected)")
    else:
        if checkpoint:
            log.warning(f"  checkpoint not found: {checkpoint}  (using pretrained only)")

    model.eval()
    return model.to(device) if device is not None else model


@torch.no_grad()
def embed_images(model: nn.Module,
                 top_n: np.ndarray, bot_n: np.ndarray, depth: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """
    Pack three [H, W, 3] uint8 renders into a single [1, 7, H, W] float tensor,
    normalise to [-1, 1], and return the 384-d CLS-token embedding.
    """
    top_f = top_n.astype(np.float32) / 255.0            # [H, W, 3]
    bot_f = bot_n.astype(np.float32) / 255.0            # [H, W, 3]
    dep_f = depth[:, :, :1].astype(np.float32) / 255.0  # [H, W, 1]
    arr7  = np.concatenate([top_f, bot_f, dep_f], axis=2)  # [H, W, 7]
    t = torch.from_numpy(arr7.transpose(2, 0, 1)).unsqueeze(0).to(device)  # [1, 7, H, W]
    t = (t - 0.5) / 0.5
    return model(t).cpu().numpy()[0].astype(np.float32)  # [D]


# ---------------------------------------------------------------------------
# Plot helpers (DINOv2-titled, shared utilities from demo_embed_pca)
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
    p.add_argument("--wrl_dir",        default=str(WRL_DIR))
    p.add_argument("--output_dir",     default=str(OUTPUT_DIR))
    p.add_argument("--site_xlsx",      default=str(XLSX_DEFAULT),
                   help="Excel file mapping WRL stems to sites (optional)")
    p.add_argument("--no_site_color",  action="store_true")
    p.add_argument("--dino_model",     default=DINO_MODEL,
                   help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14")
    p.add_argument("--image_size",     type=int, default=IMAGE_SIZE,
                   help="Render canvas size in pixels (multiple of 14)")
    p.add_argument("--device",         default=None)
    p.add_argument("--save_renders",   action="store_true",
                   help="Save rendered PNG images to outputs/dino_7ch/renders/")
    p.add_argument("--limit",          type=int, default=None,
                   help="Process only first N WRL files")
    # Cleaning
    p.add_argument("--skip_clean",     action="store_true",
                   help="Skip mesh cleaning (use raw geometry)")
    # Decimation
    p.add_argument("--decimate",       action="store_true",
                   help="Decimate mesh before rendering (speeds up rasterisation)")
    p.add_argument("--target_faces",   type=int, default=None,
                   help="Fixed face count after decimation")
    p.add_argument("--target_ratio",   type=float, default=0.05,
                   help="Fraction of faces to keep when --target_faces is not set")
    p.add_argument("--decimate_method", default="qem", choices=["qem", "cluster"])
    # Misc
    p.add_argument("--pca_only",       action="store_true",
                   help="Skip rendering; reload cached embeddings and replot")
    p.add_argument("--seed",           type=int, default=42)
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
    """Load → clean → decimate a mesh. Returns (vertices, faces) or None on failure."""
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
            f = orient_normals_outward(v, f)
            log.info(f"  decimate {len(v):>8,}V  {len(f):>8,}F  (target {tf})")
        except Exception as e:
            log.warning(f"  Decimate failed ({e}), continuing with full mesh")

    return v, f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    device  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out     = ensure_dir(Path(args.output_dir))
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

    # Pass 1: determine global scale so all objects share the same px/unit ratio
    global_scale = find_global_scale(files, args.image_size, MARGIN)

    # Load 7-channel DINOv2
    model = build_dino7ch(args.dino_model, device=device)
    D     = model.embed_dim
    log.info(f"DINOv2-7ch {args.dino_model}  embed_dim={D}  device={device}")
    log.info(f"Device: {device}  |  image_size: {args.image_size}  |  "
             f"skip_clean={args.skip_clean}  decimate={args.decimate}")

    if args.save_renders:
        (out / "renders").mkdir(parents=True, exist_ok=True)

    stems:      list[str]        = []
    embeddings: list[np.ndarray] = []

    # Pass 2: render + embed
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

                top_n, bot_n, depth = render_views(v, f, scale=global_scale,
                                                   img_size=args.image_size)

                if args.save_renders:
                    rd = out / "renders"
                    Image.fromarray(top_n).save(rd / f"{stem}_top.png")
                    Image.fromarray(bot_n).save(rd / f"{stem}_bot.png")
                    Image.fromarray(depth).save(rd / f"{stem}_depth.png")

                emb = embed_images(model, top_n, bot_n, depth, device)  # [D]
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
