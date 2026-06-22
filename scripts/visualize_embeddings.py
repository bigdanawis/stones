"""
Visualize Point-MAE embeddings: PCA, UMAP, nearest-neighbour table.

Usage
-----
python scripts/visualize_embeddings.py \
    --embeddings outputs/embeddings/all_embeddings.npy \
    --metadata   outputs/embeddings/metadata.csv \
    --output_dir outputs/viz

Optional: --umap   (requires `pip install umap-learn`)
          --n_neighbors 5  (for NN table)
          --color_by <column>  (any metadata CSV column)
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ensure_dir, get_logger

log = get_logger("visualize")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(emb_path: str, meta_path: str | None):
    E = np.load(emb_path)                         # [N, D]

    if meta_path and Path(meta_path).exists():
        meta = pd.read_csv(meta_path)
        # Try to align by row order
        labels = meta["filename"].tolist() if "filename" in meta.columns else [str(i) for i in range(len(E))]
    else:
        meta = None
        labels = [str(i) for i in range(len(E))]

    if len(labels) != len(E):
        log.warning(f"Label count ({len(labels)}) ≠ embedding count ({len(E)}); using indices.")
        labels = [str(i) for i in range(len(E))]

    return E, labels, meta


def _scatter(ax, coords, labels, color_vals, cmap, title):
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=color_vals, cmap=cmap, s=40, alpha=0.8, edgecolors="none",
    )
    for i, lbl in enumerate(labels):
        # only annotate if not too many points
        if len(labels) <= 60:
            ax.annotate(
                Path(lbl).stem[:12],
                (coords[i, 0], coords[i, 1]),
                fontsize=6, alpha=0.7,
            )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.axis("equal")
    return sc


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def plot_pca(E, labels, meta, color_col, out_dir):
    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E)
    pca = PCA(n_components=min(50, E.shape[0], E.shape[1]))
    pca.fit(E_scaled)
    coords = pca.transform(E_scaled)[:, :2]

    var = pca.explained_variance_ratio_
    log.info(f"PCA: PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2] if len(var)>2 else 0:.1%}")

    color_vals, cmap = _get_color(meta, labels, color_col)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = _scatter(axes[0], coords, labels, color_vals, cmap,
                  f"PCA  ({var[0]:.1%} + {var[1]:.1%})")
    if color_vals is not None:
        plt.colorbar(sc, ax=axes[0], label=color_col or "index")

    # Scree plot
    axes[1].plot(np.arange(1, len(var) + 1), np.cumsum(var), marker="o", markersize=4)
    axes[1].axhline(0.9, linestyle="--", color="red", alpha=0.6, label="90%")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title("PCA scree")
    axes[1].legend()

    plt.tight_layout()
    out = out_dir / "pca.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"PCA plot → {out}")

    # Save PCA coordinates
    df_pca = pd.DataFrame(pca.transform(E_scaled), columns=[f"pc{i+1}" for i in range(pca.n_components_)])
    df_pca.insert(0, "filename", labels)
    df_pca.to_csv(out_dir / "pca_coords.csv", index=False)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def plot_umap(E, labels, meta, color_col, out_dir, n_neighbors=15, min_dist=0.1):
    try:
        import umap  # noqa: PLC0415
    except ImportError:
        log.warning("umap-learn not installed. Run: pip install umap-learn")
        return

    log.info("Running UMAP…")
    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(E_scaled)

    color_vals, cmap = _get_color(meta, labels, color_col)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = _scatter(ax, coords, labels, color_vals, cmap, "UMAP")
    if color_vals is not None:
        plt.colorbar(sc, ax=ax, label=color_col or "index")
    plt.tight_layout()
    out = out_dir / "umap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"UMAP plot → {out}")

    df_umap = pd.DataFrame(coords, columns=["u1", "u2"])
    df_umap.insert(0, "filename", labels)
    df_umap.to_csv(out_dir / "umap_coords.csv", index=False)


# ---------------------------------------------------------------------------
# Nearest neighbours
# ---------------------------------------------------------------------------

def nearest_neighbors(E, labels, n_neighbors, out_dir):
    k = min(n_neighbors + 1, len(E))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(E)
    distances, indices = nn.kneighbors(E)

    rows = []
    for i, lbl in enumerate(labels):
        for rank, (j, d) in enumerate(zip(indices[i, 1:], distances[i, 1:]), 1):
            rows.append({
                "query":    Path(lbl).stem,
                "rank":     rank,
                "neighbor": Path(labels[j]).stem,
                "cosine_distance": round(float(d), 5),
            })

    df = pd.DataFrame(rows)
    out = out_dir / "nearest_neighbors.csv"
    df.to_csv(out, index=False)
    log.info(f"Nearest-neighbour table → {out}")

    # Print top-1 neighbour for each object
    top1 = df[df["rank"] == 1][["query", "neighbor", "cosine_distance"]]
    log.info(f"\n{top1.to_string(index=False)}")


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def _get_color(meta, labels, color_col):
    if meta is None or color_col is None or color_col not in meta.columns:
        color_vals = np.arange(len(labels))
        cmap = "viridis"
    else:
        col = meta.set_index("filename")[color_col].reindex(labels)
        try:
            color_vals = col.astype(float).to_numpy()
        except Exception:
            # Categorical → integer codes
            color_vals = col.astype("category").cat.codes.to_numpy().astype(float)
        cmap = "tab20"
    return color_vals, cmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Point-MAE embeddings.")
    parser.add_argument("--embeddings",  required=True, help="Path to all_embeddings.npy")
    parser.add_argument("--metadata",    default=None,  help="Path to metadata.csv")
    parser.add_argument("--output_dir",  default="outputs/viz")
    parser.add_argument("--umap",        action="store_true", help="Run UMAP (needs umap-learn)")
    parser.add_argument("--n_neighbors", type=int, default=5, help="k for nearest-neighbour table")
    parser.add_argument("--color_by",    default=None, help="Metadata column to use for colouring")
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist",  type=float, default=0.1)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.output_dir))
    E, labels, meta = load_embeddings(args.embeddings, args.metadata)

    log.info(f"Loaded {len(E)} embeddings of dim {E.shape[1]}")

    if len(E) < 2:
        log.error("Need at least 2 embeddings to visualize.")
        sys.exit(1)

    plot_pca(E, labels, meta, args.color_by, out_dir)
    nearest_neighbors(E, labels, args.n_neighbors, out_dir)

    if args.umap:
        plot_umap(E, labels, meta, args.color_by, out_dir,
                  n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)

    log.info(f"All outputs in {out_dir}")


if __name__ == "__main__":
    main()
