"""
Convert saved 7-channel .npy renders to human-viewable images.

For each {stem}.npy in the renders/ directory, saves three images alongside:
  {stem}_top.png   — RGB top-view normal map    (ch 0-2, outward normals)
  {stem}_bot.png   — RGB bottom-view normal map (ch 3-5, outward normals)
  {stem}_depth.png — grayscale top-down depth   (ch 6)

Usage:
    python scripts/export_render_images.py
    python scripts/export_render_images.py --renders_dir outputs/dino7ch/renders
    python scripts/export_render_images.py --renders_dir outputs/dino7ch/renders --limit 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_RENDERS = ROOT / "outputs" / "dino7ch" / "renders"


def npy_to_images(npy_path: Path, force: bool = False):
    stem = npy_path.stem
    d = npy_path.parent

    top_path   = d / f"{stem}_top.png"
    bot_path   = d / f"{stem}_bot.png"
    depth_path = d / f"{stem}_depth.png"

    if not force and top_path.exists() and bot_path.exists() and depth_path.exists():
        return  # already exported

    arr = np.load(npy_path)   # [H, W, 7] float32 in [0, 1]

    top_rgb   = (arr[:, :, 0:3] * 255).clip(0, 255).astype(np.uint8)
    bot_rgb   = (arr[:, :, 3:6] * 255).clip(0, 255).astype(np.uint8)
    depth_gray = (arr[:, :, 6]  * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(top_rgb).save(top_path)
    Image.fromarray(bot_rgb).save(bot_path)
    Image.fromarray(depth_gray, mode="L").save(depth_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--renders_dir", default=str(DEFAULT_RENDERS),
                   help="Directory containing .npy render files")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing images")
    args = p.parse_args()

    renders_dir = Path(args.renders_dir)
    if not renders_dir.is_dir():
        print(f"Renders directory not found: {renders_dir}")
        sys.exit(1)

    files = sorted(renders_dir.glob("*.npy"))
    if args.limit:
        files = files[: args.limit]

    if not files:
        print(f"No .npy files found in {renders_dir}")
        sys.exit(1)

    print(f"Exporting {len(files)} renders from {renders_dir}")
    for f in tqdm(files, unit="mesh"):
        npy_to_images(f, force=args.force)

    print(f"Done. Images saved alongside .npy files in {renders_dir}")


if __name__ == "__main__":
    main()
