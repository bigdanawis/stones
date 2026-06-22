"""
Main pipeline: WRL meshes → Point-MAE embeddings.

Usage
-----
python scripts/extract_pointmae_embeddings.py \
    --input_dir  data/wrl \
    --output_dir outputs/embeddings \
    --checkpoint path/to/pointmae.pth \
    --num_points 4096 \
    --sampling_mode edge_aware \
    --device cuda

Set POINTMAE_REPO=/path/to/Point-MAE-repo to use the official model.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mesh_cleaning import (
    clean_mesh,
    compute_surface_area,
)
from src.mesh_io import load_mesh
from src.pointmae_embedder import build_embedder
from src.sampling import sample_points
from src.utils import (
    FailureWriter,
    MetadataWriter,
    Timer,
    collect_mesh_files,
    ensure_dir,
    get_logger,
)

log = get_logger("pipeline")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_mesh(vertices: np.ndarray):
    """
    Center at origin, scale by bounding-box diagonal.
    Returns (vertices_norm, centroid, scale, bbox).
    """
    centroid = vertices.mean(axis=0)
    v = vertices - centroid
    bbox = v.max(axis=0) - v.min(axis=0)
    diagonal = float(np.linalg.norm(bbox))
    scale = diagonal if diagonal > 0 else 1.0
    v /= scale
    return v, centroid, scale, bbox


def save_ply_pointcloud(path: Path, points: np.ndarray, normals: np.ndarray):
    """Write a minimal ASCII PLY for quick inspection."""
    N = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            nx, ny, nz = normals[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_file(
    mesh_path: Path,
    output_dir: Path,
    embedder,
    num_points: int,
    sampling_mode: str,
    edge_ratio: float,
    save_previews: bool,
    keep_largest_component: bool,
    skip_clean: bool,
    meta_writer: MetadataWriter,
    fail_writer: FailureWriter,
    rng: np.random.Generator,
):
    stem = mesh_path.stem
    timer = Timer()

    # ---- 1. Load ----
    try:
        raw = load_mesh(mesh_path)
    except Exception as e:
        log.error(f"LOAD FAILED  {mesh_path.name}: {e}")
        fail_writer.write(mesh_path.name, "load", str(e))
        return

    vertices_raw = raw["vertices"]
    faces_raw    = raw["faces"]
    n_orig_v     = len(vertices_raw)
    n_orig_f     = len(faces_raw)
    log.info(f"[{stem}] loaded ({n_orig_v:,}V / {n_orig_f:,}F) via {raw['source']}")

    # ---- 2. Clean ----
    if skip_clean:
        log.info(f"[{stem}] cleaning skipped (--skip_clean)")
        vertices_clean, faces_clean = vertices_raw, faces_raw
        clean_meta = {
            "num_clean_vertices": len(vertices_raw),
            "num_clean_faces":    len(faces_raw),
        }
    else:
        try:
            vertices_clean, faces_clean, clean_meta = clean_mesh(
                vertices_raw, faces_raw,
                keep_largest_component=keep_largest_component,
            )
        except Exception as e:
            log.error(f"CLEAN FAILED {mesh_path.name}: {e}")
            fail_writer.write(mesh_path.name, "clean", str(e))
            return

        if len(faces_clean) == 0:
            msg = "No faces remain after cleaning."
            log.error(f"CLEAN FAILED {mesh_path.name}: {msg}")
            fail_writer.write(mesh_path.name, "clean", msg)
            return

    # ---- 3. Normalize ----
    vertices_norm, centroid, scale, bbox = normalize_mesh(vertices_clean)
    surface_area = compute_surface_area(vertices_clean, faces_clean)

    # ---- 4. Sample ----
    try:
        points, normals = sample_points(
            vertices_norm, faces_clean, num_points,
            mode=sampling_mode, edge_ratio=edge_ratio, rng=rng,
        )
    except Exception as e:
        log.error(f"SAMPLE FAILED {mesh_path.name}: {e}")
        fail_writer.write(mesh_path.name, "sample", str(e))
        return

    # ---- 5. Embed ----
    try:
        embedding = embedder.embed(points)
    except Exception as e:
        log.error(f"EMBED FAILED {mesh_path.name}: {e}")
        fail_writer.write(mesh_path.name, "embed", str(e))
        return

    # ---- 6. Save ----
    pc_path  = output_dir / "pointclouds" / f"{stem}.npy"
    emb_path = output_dir / "embeddings"  / f"{stem}.npy"
    ensure_dir(pc_path.parent)
    ensure_dir(emb_path.parent)

    np.save(pc_path,  np.concatenate([points, normals], axis=1))
    np.save(emb_path, embedding)
    np.savetxt(emb_path.with_suffix(".csv"), embedding[np.newaxis], delimiter=",")

    preview_path = ""
    if save_previews:
        ply_path = output_dir / "previews" / f"{stem}.ply"
        ensure_dir(ply_path.parent)
        save_ply_pointcloud(ply_path, points, normals)
        preview_path = str(ply_path)

    elapsed = timer.elapsed()
    log.info(f"[{stem}] done in {elapsed:.1f}s  embedding dim={len(embedding)}")

    meta_writer.write({
        "filename":              mesh_path.name,
        "num_original_vertices": n_orig_v,
        "num_original_faces":    n_orig_f,
        "num_clean_vertices":    clean_meta["num_clean_vertices"],
        "num_clean_faces":       clean_meta["num_clean_faces"],
        "sampling_mode":         sampling_mode,
        "num_points":            num_points,
        "centroid_x":            centroid[0],
        "centroid_y":            centroid[1],
        "centroid_z":            centroid[2],
        "scale_factor":          scale,
        "bbox_x":                bbox[0],
        "bbox_y":                bbox[1],
        "bbox_z":                bbox[2],
        "surface_area":          surface_area,
        "embedding_path":        str(emb_path),
        "pointcloud_path":       str(pc_path),
        "preview_path":          preview_path,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract Point-MAE embeddings from WRL meshes.")
    p.add_argument("--input_dir",     required=True,  help="Directory containing .wrl/.vrml files")
    p.add_argument("--output_dir",    required=True,  help="Root directory for outputs")
    p.add_argument("--checkpoint",    default=None,   help="Path to Point-MAE pretrained checkpoint")
    p.add_argument("--num_points",    type=int, default=4096,
                   help="Number of surface points to sample (default 4096)")
    p.add_argument("--sampling_mode", default="edge_aware",
                   choices=["uniform", "curvature", "edge_aware"],
                   help="Point sampling strategy")
    p.add_argument("--edge_ratio",    type=float, default=0.30,
                   help="Fraction of points from high-dihedral zone in edge_aware mode")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit",         type=int, default=None,
                   help="Process at most N files (for testing)")
    p.add_argument("--save_previews", action="store_true",
                   help="Save PLY point cloud preview for each mesh")
    p.add_argument("--keep_largest",  action="store_true", default=True,
                   help="Keep only the largest connected component (default True)")
    p.add_argument("--skip_clean",    action="store_true",
                   help="Skip mesh cleaning entirely (use raw loaded geometry)")
    p.add_argument("--embed_dim",     type=int, default=384,
                   help="Encoder embedding dimension (default 384 = ViT-Small)")
    p.add_argument("--num_groups",    type=int, default=64,
                   help="Number of point groups for Point-MAE (default 64)")
    p.add_argument("--group_size",    type=int, default=32,
                   help="Points per group (K) for Point-MAE (default 32)")
    p.add_argument("--seed",          type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out = ensure_dir(Path(args.output_dir))
    meta_path = out / "metadata.csv"
    fail_path = out / "failed_meshes.csv"
    meta_writer = MetadataWriter(meta_path)
    fail_writer = FailureWriter(fail_path)

    log.info(f"Input : {args.input_dir}")
    log.info(f"Output: {out}")
    log.info(f"Device: {args.device}  |  points: {args.num_points}  |  mode: {args.sampling_mode}")

    mesh_files = collect_mesh_files(args.input_dir, limit=args.limit)
    if not mesh_files:
        log.error("No .wrl/.vrml files found in input directory.")
        sys.exit(1)
    log.info(f"Found {len(mesh_files)} mesh file(s)")

    embedder = build_embedder(
        checkpoint=args.checkpoint,
        device=args.device,
        num_points=args.num_points,
        group_size=args.group_size,
        num_groups=args.num_groups,
        embed_dim=args.embed_dim,
    )

    for i, mesh_path in enumerate(mesh_files, 1):
        log.info(f"--- [{i}/{len(mesh_files)}] {mesh_path.name}")
        try:
            process_file(
                mesh_path=mesh_path,
                output_dir=out,
                embedder=embedder,
                num_points=args.num_points,
                sampling_mode=args.sampling_mode,
                edge_ratio=args.edge_ratio,
                save_previews=args.save_previews,
                keep_largest_component=args.keep_largest,
                skip_clean=args.skip_clean,
                meta_writer=meta_writer,
                fail_writer=fail_writer,
                rng=rng,
            )
        except Exception:
            log.error(f"Unhandled exception for {mesh_path.name}:\n{traceback.format_exc()}")
            fail_writer.write(mesh_path.name, "pipeline", traceback.format_exc()[-200:])

    log.info(f"Done. Metadata → {meta_path}  |  Failures → {fail_path}")

    # Consolidate all embeddings into a single file for convenience
    _consolidate_embeddings(out)


def _consolidate_embeddings(out: Path):
    emb_dir = out / "embeddings"
    if not emb_dir.exists():
        return
    files = sorted(emb_dir.glob("*.npy"))
    if not files:
        return
    names = [f.stem for f in files]
    arrays = [np.load(f) for f in files]
    matrix = np.stack(arrays, axis=0)
    np.save(out / "all_embeddings.npy", matrix)

    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.DataFrame(matrix, index=names)
        df.index.name = "filename"
        df.to_csv(out / "all_embeddings.csv", index_label="stem")
        df.to_parquet(out / "all_embeddings.parquet")
        log.info(f"Consolidated {len(files)} embeddings -> all_embeddings.csv + .parquet")
    except Exception as e:
        log.warning(f"Could not write parquet ({e}); all_embeddings.npy saved instead.")


if __name__ == "__main__":
    main()
