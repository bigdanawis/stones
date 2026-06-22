"""
Diagnostic script — quickly test mesh loading and cleaning on one or more files.

Usage
-----
python scripts/check_mesh_loading.py path/to/file.wrl [path/to/another.wrl ...]
python scripts/check_mesh_loading.py --dir data/wrl --limit 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mesh_cleaning import (
    clean_mesh,
    compute_surface_area,
)
from src.mesh_io import load_mesh
from src.sampling import sample_points
from src.utils import collect_mesh_files, get_logger

log = get_logger("check_mesh_loading")

LINE = "-" * 60


def check_file(path: Path, num_points: int = 2048, mode: str = "edge_aware"):
    print(LINE)
    print(f"File : {path}")
    print(f"Size : {path.stat().st_size / 1e6:.1f} MB")

    # Load
    try:
        raw = load_mesh(path)
    except Exception as e:
        print(f"[FAIL] Load: {e}")
        return

    v, f = raw["vertices"], raw["faces"]
    print(f"Loader : {raw['source']}")
    print(f"Raw    : {len(v):,} vertices  /  {len(f):,} faces")

    if len(v) == 0 or len(f) == 0:
        print("[FAIL] Empty mesh after loading.")
        return

    bbox = v.max(axis=0) - v.min(axis=0)
    print(f"BBox   : {bbox[0]:.3f} × {bbox[1]:.3f} × {bbox[2]:.3f}")

    # Clean
    try:
        vc, fc, meta = clean_mesh(v, f, keep_largest_component=True)
    except Exception as e:
        print(f"[FAIL] Clean: {e}")
        return
    print(f"Clean  : {len(vc):,} vertices  /  {len(fc):,} faces")
    area = compute_surface_area(vc, fc)
    print(f"Area   : {area:.4f} (in mesh units²)")

    if len(fc) == 0:
        print("[FAIL] No faces after cleaning.")
        return

    # Normalize
    centroid = vc.mean(axis=0)
    vn = vc - centroid
    diag = float(np.linalg.norm(vn.max(axis=0) - vn.min(axis=0)))
    if diag > 0:
        vn /= diag
    print(f"Normalized extent: {np.abs(vn).max():.4f}")

    # Sample
    for smode in ["uniform", "curvature", "edge_aware"]:
        try:
            pts, nrm = sample_points(vn, fc, num_points, mode=smode,
                                     rng=np.random.default_rng(0))
            normals_ok = not np.isnan(nrm).any() and not np.isinf(nrm).any()
            print(f"Sample [{smode:12s}]: pts {pts.shape}  normals_ok={normals_ok}")
        except Exception as e:
            print(f"Sample [{smode:12s}]: FAILED — {e}")

    print("[OK]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="One or more .wrl files")
    parser.add_argument("--dir",    default=None, help="Directory to scan for .wrl files")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--points", type=int, default=2048)
    args = parser.parse_args()

    paths = [Path(p) for p in args.files]
    if args.dir:
        paths += collect_mesh_files(args.dir, limit=args.limit)
    if not paths:
        print("No files specified. Use positional args or --dir.")
        sys.exit(1)

    for p in paths:
        check_file(p, num_points=args.points)
    print(LINE)
    print(f"Checked {len(paths)} file(s).")


if __name__ == "__main__":
    main()
