"""
Regenerate PCA and UMAP plots with an updated palette, saving as *_updated.png.
Reads all_embeddings.npy + embeddings/*.npy from each target directory.

Usage:
    python scripts/replot_updated_palette.py
    python scripts/replot_updated_palette.py --dirs outputs/dino7ch/base outputs/dino7ch/finetuned
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Patch palette BEFORE importing anything that reads _PALETTE ──────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import scripts.demo_embed_pca as _pca_mod

_pca_mod._PALETTE = [
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

import scripts.demo_dino7ch_embed as _dino_mod

# Patch the palette reference inside the dino module too
_dino_mod._PALETTE = _pca_mod._PALETTE

from scripts.demo_embed_pca import load_site_map
from scripts.demo_dino7ch_embed import plot_pca, plot_umap, _load_site_and_notes
from src.utils import get_logger

log = get_logger("replot")

XLSX_DEFAULT = ROOT / "wrl" / "Handaxes 2026 list with sites.xlsx"
DEFAULT_DIRS = [
    ROOT / "outputs" / "dino7ch" / "base",
    ROOT / "outputs" / "dino7ch" / "finetuned",
]


def load_embeddings(d: Path):
    emb_dir = d / "embeddings"
    files = sorted(emb_dir.glob("*.npy")) if emb_dir.is_dir() else []
    if not files:
        log.warning(f"No embeddings found in {emb_dir}")
        return [], []
    stems = [f.stem for f in files]
    embeddings = [np.load(f) for f in files]
    return stems, embeddings


def _patched_plot(stems, embeddings, out, site_map, notes_map, label):
    """Save pca_updated.png and umap_updated.png into out/."""
    # Temporarily redirect output filenames by monkey-patching _plot_2d
    original_plot_2d = _dino_mod._plot_2d

    def patched_plot_2d(stems, coords, site_map, notes_map, title, xlabel, ylabel, out_path):
        updated_path = out_path.parent / (out_path.stem + "_updated" + out_path.suffix)
        original_plot_2d(stems, coords, site_map, notes_map, title, xlabel, ylabel, updated_path)

    _dino_mod._plot_2d = patched_plot_2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plot_pca(stems, embeddings, out, site_map, notes_map)
            plot_umap(stems, embeddings, out, site_map, notes_map)
            log.info(f"{label}: saved pca_updated.png and umap_updated.png → {out}")
        finally:
            _dino_mod._plot_2d = original_plot_2d


_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p', '<', '>', 'H', '8', 'd']


def _plot_2d_markers(stems, coords, site_map, notes_map, title, xlabel, ylabel, out_path):
    """Like _plot_2d but assigns a distinct marker per site in addition to color."""
    from scripts.demo_embed_pca import _site_colors
    stem_sites, unique_sites, site_color = _site_colors(stems, site_map)
    site_marker = {s: _MARKERS[i % len(_MARKERS)] for i, s in enumerate(unique_sites)}

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    if site_map:
        for site in unique_sites:
            idxs = [i for i, s in enumerate(stem_sites) if s == site]
            ax.scatter(coords[idxs, 0], coords[idxs, 1],
                       color=site_color[site], marker=site_marker[site],
                       s=120, zorder=3, edgecolors="white", linewidths=0.5, label=site)
        leg = ax.legend(
            title="Site", title_fontsize=8, fontsize=7,
            facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white",
            framealpha=0.85, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        )
        leg.get_title().set_color("#aaaaaa")
    else:
        palette = _pca_mod._PALETTE
        for i, (x, y) in enumerate(coords[:, :2]):
            ax.scatter(x, y, color=palette[i % len(palette)],
                       marker=_MARKERS[i % len(_MARKERS)],
                       s=120, zorder=3, edgecolors="white", linewidths=0.5)

    if notes_map:
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


def _patched_plot_markers(stems, embeddings, out, site_map, notes_map, label):
    """Save pca_updated_mrkr.png and umap_updated_mrkr.png using color + marker per site."""
    original_plot_2d = _dino_mod._plot_2d

    def patched(stems, coords, site_map, notes_map, title, xlabel, ylabel, out_path):
        mrkr_path = out_path.parent / (out_path.stem + "_updated_mrkr" + out_path.suffix)
        _plot_2d_markers(stems, coords, site_map, notes_map, title, xlabel, ylabel, mrkr_path)

    _dino_mod._plot_2d = patched
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plot_pca(stems, embeddings, out, site_map, notes_map)
            plot_umap(stems, embeddings, out, site_map, notes_map)
            log.info(f"{label}: saved pca_updated_mrkr.png and umap_updated_mrkr.png")
        finally:
            _dino_mod._plot_2d = original_plot_2d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=None,
                   help="Directories containing embeddings/ subdir (default: base + finetuned)")
    p.add_argument("--site_xlsx", default=str(XLSX_DEFAULT))
    p.add_argument("--no_site_color", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dirs = [Path(d) for d in args.dirs] if args.dirs else DEFAULT_DIRS

    site_map, notes_map = None, None
    if not args.no_site_color:
        xlsx = Path(args.site_xlsx)
        if xlsx.exists():
            try:
                site_map, notes_map = load_site_map(xlsx)
            except Exception as e:
                log.warning(f"Could not load site map: {e}")

    for d in dirs:
        if not d.is_dir():
            log.warning(f"Directory not found, skipping: {d}")
            continue
        stems, embeddings = load_embeddings(d)
        if not stems:
            continue
        log.info(f"{d.name}: {len(stems)} embeddings, dim={embeddings[0].shape[0]}")
        _patched_plot(stems, embeddings, d, site_map, notes_map, label=d.name)
        _patched_plot_markers(stems, embeddings, d, site_map, notes_map, label=d.name)


if __name__ == "__main__":
    main()
