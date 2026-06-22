"""
Mesh loading with layered fallbacks:
  1. trimesh (if installed)
  2. meshio  (if installed)
  3. Built-in VRML/WRL parser (always available)

Returns a dict:
  vertices : np.ndarray  [V, 3]  float64
  faces    : np.ndarray  [F, 3]  int32
  source   : str         loader that succeeded
"""
import re
import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_mesh(path: str | Path) -> Dict:
    """
    Load a WRL/VRML file and return vertices + triangular faces.
    Raises RuntimeError if all loaders fail.
    """
    path = Path(path)
    errors = []

    for loader_name, loader_fn in [
        ("trimesh", _load_trimesh),
        ("meshio",  _load_meshio),
        ("vrml",    _load_vrml_native),
    ]:
        try:
            result = loader_fn(path)
            if result is not None and len(result["vertices"]) > 0 and len(result["faces"]) > 0:
                result["source"] = loader_name
                log.debug(f"{path.name}: loaded via {loader_name} "
                          f"({len(result['vertices']):,}V / {len(result['faces']):,}F)")
                return result
        except Exception as e:
            errors.append(f"{loader_name}: {e}")
            log.debug(f"{path.name}: {loader_name} failed — {e}")

    raise RuntimeError(f"All loaders failed for {path.name}: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Loader 1 — trimesh
# ---------------------------------------------------------------------------

def _load_trimesh(path: Path) -> Optional[Dict]:
    import trimesh  # noqa: PLC0415
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    return {
        "vertices": np.array(mesh.vertices, dtype=np.float64),
        "faces":    np.array(mesh.faces,    dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Loader 2 — meshio
# ---------------------------------------------------------------------------

def _load_meshio(path: Path) -> Optional[Dict]:
    import meshio  # noqa: PLC0415
    m = meshio.read(str(path))
    verts = np.array(m.points, dtype=np.float64)
    faces = []
    for cell_block in m.cells:
        if cell_block.type == "triangle":
            faces.append(cell_block.data.astype(np.int32))
        elif cell_block.type == "quad":
            # split quads into two triangles
            q = cell_block.data
            faces.append(np.stack([q[:, 0], q[:, 1], q[:, 2]], axis=1).astype(np.int32))
            faces.append(np.stack([q[:, 0], q[:, 2], q[:, 3]], axis=1).astype(np.int32))
    if not faces:
        return None
    return {
        "vertices": verts,
        "faces":    np.concatenate(faces, axis=0),
    }


# ---------------------------------------------------------------------------
# Loader 3 — native VRML/WRL parser
# ---------------------------------------------------------------------------

def _load_vrml_native(path: Path) -> Optional[Dict]:
    """
    Hand-written parser for VRML 2.0 / X3D files produced by 3D scanners.
    Handles the most common structure:
      Shape { geometry IndexedFaceSet { coord Coordinate { point [...] }
                                        coordIndex [...] } }
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    vertices = _parse_vrml_points(text)
    faces    = _parse_vrml_indices(text, len(vertices))
    if vertices is None or faces is None:
        return None
    return {"vertices": vertices, "faces": faces}


def _parse_vrml_points(text: str) -> Optional[np.ndarray]:
    # Find the block between "point [" and the matching "]"
    m = re.search(r'\bpoint\s*\[', text, re.IGNORECASE)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
        i += 1
    block = text[start:i - 1]

    # Strip comments and tokenise
    block = re.sub(r'#[^\n]*', '', block)
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block)
    if len(nums) < 3:
        return None
    n_verts = len(nums) // 3
    arr = np.array(nums[:n_verts * 3], dtype=np.float64).reshape(n_verts, 3)
    return arr


def _parse_vrml_indices(text: str, n_verts: int) -> Optional[np.ndarray]:
    m = re.search(r'\bcoordIndex\s*\[', text, re.IGNORECASE)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
        i += 1
    block = text[start:i - 1]
    block = re.sub(r'#[^\n]*', '', block)

    nums = np.fromiter(
        (int(x) for x in re.findall(r'-?\d+', block)),
        dtype=np.int32,
    )

    # Split on -1 sentinels; keep only triangles / quads
    faces = []
    buf = []
    for idx in nums:
        if idx == -1:
            if len(buf) == 3:
                faces.append(buf)
            elif len(buf) == 4:
                # fan-triangulate quad
                faces.append([buf[0], buf[1], buf[2]])
                faces.append([buf[0], buf[2], buf[3]])
            buf = []
        else:
            buf.append(idx)
    # handle last face without trailing -1
    if len(buf) == 3:
        faces.append(buf)

    if not faces:
        return None

    arr = np.array(faces, dtype=np.int32)
    # Clamp out-of-range indices
    valid = np.all((arr >= 0) & (arr < n_verts), axis=1)
    arr = arr[valid]
    return arr if len(arr) > 0 else None
