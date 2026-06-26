"""
Correlate PCA components with image-derived statistics.

For each stone, computes from its render PNGs:
  n_pixels        : number of non-zero (mask) pixels
  thick_mean      : mean thickness  in masked area
  thick_std       : std  thickness  in masked area
  thick_coverage  : fraction of mask where thickness > 0  (both surfaces visible)
  dih_mean        : mean dihedral   in masked area
  dih_std         : std  dihedral   in masked area
  dih_coverage    : fraction of mask where dihedral > 0

Shape statistics (from silhouette mask):
  aspect_ratio    : sqrt(λ1/λ2) of mask pixel PCA — elongation (1=circle, >1=elongated)
  circularity     : 4π·area/perimeter² — how round the outline is (1=circle, <1=irregular)
  solidity        : area / convex-hull area — convexity of the outline
  tip_sharpness   : width at the pointier tip / max width — lower = more pointed

Then prints Pearson r (and p-value) of each stat vs. PC1..N, and
saves a grid of scatter plots.

Usage
-----
  python scripts/pc_correlation.py
  python scripts/pc_correlation.py --run_dir outputs/dino_7ch_v2/20260624_123720
  python scripts/pc_correlation.py --renders_dir outputs/renders_v3 --out_png pc_corr.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_BASE = ROOT / "outputs" / "dino_7ch_v2"
RENDERS_DIR = ROOT / "outputs" / "renders"

MASK_THRESH = 1.5 / 255.0   # same threshold used in dino_v3.py
PIX_THRESH  = 0.5 / 255.0   # threshold for "non-zero" single-channel pixel


# ---------------------------------------------------------------------------
# Image statistics
# ---------------------------------------------------------------------------

def stone_stats(stem: str, renders_dir: Path) -> dict[str, float] | None:
    top_path  = renders_dir / f"{stem}_top.png"
    thk_path  = renders_dir / f"{stem}_thick.png"
    dih_path  = renders_dir / f"{stem}_dihedral.png"

    if not top_path.exists():
        return None

    top = np.array(Image.open(top_path),  dtype=np.float32) / 255.0  # [H,W,3]
    mask = top.sum(axis=2) > MASK_THRESH                               # [H,W] bool

    n_pix = float(mask.sum())
    if n_pix == 0:
        return None

    stats: dict[str, float] = {"n_pixels": n_pix}

    if thk_path.exists():
        thk = np.array(Image.open(thk_path), dtype=np.float32) / 255.0  # [H,W]
        thk_in = thk[mask]
        stats["thick_mean"]     = float(thk_in.mean())
        stats["thick_std"]      = float(thk_in.std())
        stats["thick_coverage"] = float((thk_in > PIX_THRESH).mean())
    else:
        stats.update(thick_mean=np.nan, thick_std=np.nan, thick_coverage=np.nan)

    if dih_path.exists():
        dih = np.array(Image.open(dih_path), dtype=np.float32) / 255.0  # [H,W]
        dih_in = dih[mask]
        stats["dih_mean"]     = float(dih_in.mean())
        stats["dih_std"]      = float(dih_in.std())
        stats["dih_coverage"] = float((dih_in > PIX_THRESH).mean())
    else:
        stats.update(dih_mean=np.nan, dih_std=np.nan, dih_coverage=np.nan)

    return stats


def shape_stats(stem: str, renders_dir: Path) -> dict[str, float] | None:
    """
    Compute plan-view shape descriptors from the silhouette mask of _top.png.

    Uses PCA on mask pixel coordinates to find the principal (long) axis,
    then measures tip width relative to max width along the perpendicular axis.
    All metrics are purely geometric — no normal-map values used.
    """
    top_path = renders_dir / f"{stem}_top.png"
    if not top_path.exists():
        return None

    top  = np.array(Image.open(top_path), dtype=np.float32) / 255.0
    mask = top.sum(axis=2) > MASK_THRESH   # [H, W] bool

    ys, xs = np.where(mask)
    n_pix  = len(ys)
    if n_pix < 20:
        return None

    # PCA on mask pixel coordinates → principal axes
    coords   = np.stack([xs, ys], axis=1).astype(np.float64)
    coords_c = coords - coords.mean(axis=0)
    cov      = (coords_c.T @ coords_c) / n_pix
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals  = eigvals[::-1]    # descending
    eigvecs  = eigvecs[:, ::-1]

    # Aspect ratio: elongation of the silhouette
    aspect_ratio = float(np.sqrt(eigvals[0] / max(eigvals[1], 1e-8)))

    # Circularity: 4π·area / perimeter²  (1 = perfect circle, <1 = irregular/elongated)
    boundary    = mask & ~binary_erosion(mask)
    perimeter   = float(boundary.sum())
    circularity = float(4 * np.pi * n_pix / max(perimeter ** 2, 1.0))

    # Solidity: area / convex-hull area  (1 = fully convex, <1 = concave outline)
    try:
        hull     = ConvexHull(coords)
        solidity = float(n_pix / max(hull.volume, 1.0))   # hull.volume = area in 2D
    except Exception:
        solidity = np.nan

    # Tip sharpness: project coords onto long axis, measure perpendicular width
    # at the 10% tip regions and compare to the maximum width.
    long_proj  = coords_c @ eigvecs[:, 0]   # projection onto long axis
    perp_proj  = coords_c @ eigvecs[:, 1]   # projection onto short axis
    p_min, p_max = long_proj.min(), long_proj.max()
    p_range    = max(p_max - p_min, 1e-8)
    tip_frac   = 0.10

    sel1 = long_proj <= (p_min + p_range * tip_frac)
    sel2 = long_proj >= (p_max - p_range * tip_frac)
    max_width = perp_proj.max() - perp_proj.min()

    def _tip_width(sel):
        p = perp_proj[sel]
        return float(p.max() - p.min()) if sel.sum() >= 3 else np.nan

    tw1, tw2    = _tip_width(sel1), _tip_width(sel2)
    tip_sharpness = (
        float(min(tw1, tw2) / max(max_width, 1e-8))
        if not (np.isnan(tw1) or np.isnan(tw2))
        else np.nan
    )

    return {
        "aspect_ratio":  aspect_ratio,
        "circularity":   circularity,
        "solidity":      solidity,
        "tip_sharpness": tip_sharpness,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAT_LABELS = {
    "n_pixels":       "# mask pixels\n(stone size)",
    "thick_mean":     "thickness mean\n(masked)",
    "thick_std":      "thickness std\n(masked)",
    "thick_coverage": "thickness coverage\n(fraction of mask > 0)",
    "dih_mean":       "dihedral mean\n(masked)",
    "dih_std":        "dihedral std\n(masked)",
    "dih_coverage":   "dihedral coverage\n(fraction of mask > 0)",
    "aspect_ratio":   "aspect ratio\n(elongation, 1=circle)",
    "circularity":    "circularity\n(4π·area/perim², 1=circle)",
    "solidity":       "solidity\n(area/convex-hull, 1=convex)",
    "tip_sharpness":  "tip sharpness\n(tip width/max width, 0=pointy)",
}


def _latest_run(base: Path) -> Path | None:
    runs = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    return runs[0] if runs else None


def _print_table(label: str, coords_v, var, S, stat_keys, valid_idx):
    n_pcs = coords_v.shape[1]
    print(f"\n{'='*70}")
    print(f"  {label}  (n={len(valid_idx)} stones)")
    print(f"{'='*70}")
    print(f"\nVariance explained: " +
          "  ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))
    print(f"\nPearson r\n")
    header = f"{'Statistic':<22}" + "".join(f"  {'PC'+str(i+1):>12}" for i in range(n_pcs))
    print(header)
    print("-" * len(header))
    for si, key in enumerate(stat_keys):
        col = S[:, si]
        valid = ~np.isnan(col)
        row = f"{key:<30}"
        for pi in range(n_pcs):
            if valid.sum() < 3:
                row += f"  {'n/a':>12}"
            else:
                r, pval = pearsonr(coords_v[valid, pi], col[valid])
                stars = ("***" if pval < 0.001 else
                         "**"  if pval < 0.01  else
                         "*"   if pval < 0.05  else "")
                row += f"  {r:+.3f}{stars:3s}  "
        print(row)
    print("\n  * p<0.05  ** p<0.01  *** p<0.001")


def _save_plot(coords_v, var, S, stat_keys, title: str, out_png: Path):
    n_pcs   = coords_v.shape[1]
    n_stats = len(stat_keys)

    fig = plt.figure(figsize=(4 * n_pcs, 3.2 * n_stats), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(n_stats, n_pcs, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.08, right=0.97, top=0.95, bottom=0.04)

    for si, key in enumerate(stat_keys):
        col = S[:, si]
        valid = ~np.isnan(col)
        for pi in range(n_pcs):
            ax = fig.add_subplot(gs[si, pi])
            ax.set_facecolor("#12122a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.tick_params(colors="#777", labelsize=7)

            x = coords_v[valid, pi]
            y = col[valid]
            ax.scatter(x, y, s=10, alpha=0.6, color="#7ba4f5", linewidths=0)

            if len(x) >= 3:
                r, pval = pearsonr(x, y)
                m, b = np.polyfit(x, y, 1)
                xl = np.array([x.min(), x.max()])
                ax.plot(xl, m * xl + b, color="#f5a742", lw=1.2, alpha=0.9)
                stars = ("***" if pval < 0.001 else
                         "**"  if pval < 0.01  else
                         "*"   if pval < 0.05  else "")
                ax.set_title(f"r={r:+.3f}{stars}  p={pval:.2e}",
                             fontsize=8, color="#ccc", pad=3)

            ax.set_xlabel(f"PC{pi+1} ({var[pi]:.1%})", fontsize=7, color="#888")
            if pi == 0:
                ax.set_ylabel(STAT_LABELS[key], fontsize=7, color="#ccc")

    fig.suptitle(title, color="#ccc", fontsize=11, y=0.975)
    fig.savefig(out_png, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_png}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir",     default=None)
    p.add_argument("--output_base", default=str(OUTPUT_BASE))
    p.add_argument("--renders_dir", default=str(RENDERS_DIR))
    p.add_argument("--n_pcs",       type=int, default=4,
                   help="How many PCs to report correlations for")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.output_base))
    if run_dir is None:
        sys.exit(f"No run folder found under {args.output_base}")
    renders_dir = Path(args.renders_dir)

    print(f"Run:     {run_dir}")
    print(f"Renders: {renders_dir}")

    # Load embeddings
    emb_files = sorted((run_dir / "embeddings").glob("*.npy"))
    if not emb_files:
        sys.exit(f"No embeddings in {run_dir / 'embeddings'}")
    stems = [f.stem for f in emb_files]
    E_raw = np.stack([np.load(f) for f in emb_files]).astype(np.float32)  # [N, D]

    # Per-stone image statistics
    print("\nComputing image statistics...")
    stat_list, valid_idx = [], []
    for i, stem in enumerate(stems):
        s = stone_stats(stem, renders_dir)
        if s is None:
            print(f"  [skip] {stem} — renders not found")
            continue
        sh = shape_stats(stem, renders_dir)
        if sh is not None:
            s.update(sh)
        stat_list.append(s)
        valid_idx.append(i)

    if not stat_list:
        sys.exit("No render images found.")

    stat_keys = list(STAT_LABELS.keys())
    S = np.array([[s.get(k, np.nan) for k in stat_keys] for s in stat_list],
                 dtype=np.float64)   # [M, n_stats]

    n_pcs = min(args.n_pcs, len(stems) - 1)

    pca      = PCA(n_components=n_pcs)
    coords   = pca.fit_transform(E_raw)
    var      = pca.explained_variance_ratio_
    coords_v = coords[valid_idx]

    _print_table("Embeddings", coords_v, var, S, stat_keys, valid_idx)
    _save_plot(
        coords_v, var, S, stat_keys,
        title="PC correlation with image statistics",
        out_png=run_dir / "pc_correlation.png",
    )


if __name__ == "__main__":
    main()
