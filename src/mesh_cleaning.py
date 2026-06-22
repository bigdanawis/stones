"""
Mesh cleaning pipeline for raw scan meshes.

Steps (in order):
  1. Remove duplicate vertices (within tolerance)
  2. Remove degenerate faces (repeated indices, zero area)
  3. Remove unreferenced vertices + re-index
  4. Fix face normals (make winding consistent via BFS)
  5. Keep largest connected component (optional)

Returns cleaned vertices, faces, and a metadata dict.
"""
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    tol: float = 1e-8,
    keep_largest_component: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Clean a triangle mesh.

    Parameters
    ----------
    vertices : [V, 3] float64
    faces    : [F, 3] int32
    tol      : distance tolerance for duplicate-vertex merging
    keep_largest_component : if True, discard disconnected pieces

    Returns
    -------
    vertices_clean, faces_clean, metadata_dict
    """
    v0, f0 = len(vertices), len(faces)
    log.info(f"Cleaning  input : {v0:>10,} V  {f0:>10,} F")

    vertices, faces = _merge_duplicate_vertices(vertices, faces, tol)
    log.info(f"  [1/5] merge duplicates : {len(vertices):>10,} V  {len(faces):>10,} F"
             f"  (removed {v0 - len(vertices):,} V)")

    faces = _remove_degenerate_faces(vertices, faces)
    log.info(f"  [2/5] drop degenerate  : {len(vertices):>10,} V  {len(faces):>10,} F"
             f"  (removed {f0 - len(faces):,} F)")

    vertices, faces = _remove_unreferenced(vertices, faces)
    log.info(f"  [3/5] drop unreferenced: {len(vertices):>10,} V  {len(faces):>10,} F")

    if keep_largest_component and len(faces) > 0:
        v_before, f_before = len(vertices), len(faces)
        vertices, faces = _keep_largest_component(vertices, faces)
        log.info(f"  [4/5] largest component: {len(vertices):>10,} V  {len(faces):>10,} F"
                 f"  (dropped {f_before - len(faces):,} F)")
    else:
        log.info(f"  [4/5] largest component: skipped")

    faces = _fix_normals(vertices, faces)
    faces = orient_normals_outward(vertices, faces)
    vertices, faces = _remove_unreferenced(vertices, faces)
    log.info(f"  [5/5] fix normals      : {len(vertices):>10,} V  {len(faces):>10,} F")

    log.info(f"Cleaning output : {len(vertices):>10,} V  {len(faces):>10,} F"
             f"  ({len(vertices)/max(v0,1)*100:.1f}% V  {len(faces)/max(f0,1)*100:.1f}% F kept)")

    meta = {
        "num_original_vertices": v0,
        "num_original_faces":    f0,
        "num_clean_vertices":    len(vertices),
        "num_clean_faces":       len(faces),
    }
    return vertices, faces, meta


# ---------------------------------------------------------------------------
# Geometry helpers used by cleaning and downstream code
# ---------------------------------------------------------------------------

def orient_normals_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Flip the winding of any face whose normal points toward the mesh centroid
    rather than away from it.  O(F), no BFS required.

    Works well for roughly convex objects (stone tools, handaxes).
    Returns a new faces array; vertices are unchanged.
    """
    if len(faces) == 0:
        return faces
    faces = faces.copy()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals       = np.cross(v1 - v0, v2 - v0)     # [F, 3] unnormalised
    face_cents    = (v0 + v1 + v2) / 3.0            # [F, 3]
    mesh_centroid = face_cents.mean(axis=0)
    outward       = face_cents - mesh_centroid       # vector from centre to face
    inward        = np.einsum("fi,fi->f", normals, outward) < 0
    faces[inward, 1], faces[inward, 2] = faces[inward, 2].copy(), faces[inward, 1].copy()
    n_flipped = int(inward.sum())
    if n_flipped:
        from src.utils import get_logger as _gl
        _gl(__name__).debug(f"orient_normals_outward: flipped {n_flipped}/{len(faces)} faces")
    return faces


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return unit normals for each face, shape [F, 3]."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    safe = length[:, 0] > 0
    n[safe] /= length[safe]
    return n.astype(np.float64)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted average of adjacent face normals, shape [V, 3]."""
    fn = compute_face_normals(vertices, faces)
    # weight each face normal by triangle area (half cross product magnitude)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1, keepdims=True) * 0.5

    vn = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn * areas)

    length = np.linalg.norm(vn, axis=1, keepdims=True)
    nonzero = length[:, 0] > 0
    vn[nonzero] /= length[nonzero]
    return vn


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return area of each triangle, shape [F]."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5


def compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    return float(compute_face_areas(vertices, faces).sum())


# ---------------------------------------------------------------------------
# Step 1 — merge duplicate vertices
# ---------------------------------------------------------------------------

def _merge_duplicate_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Round to grid, then use np.unique to deduplicate
    if tol > 0:
        scale = 1.0 / tol
        rounded = np.round(vertices * scale)
    else:
        rounded = vertices.copy()

    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    new_verts = np.array([
        vertices[np.where(inv == i)[0]].mean(axis=0)
        for i in range(inv.max() + 1)
    ], dtype=np.float64)
    new_faces = inv[faces].astype(np.int32)
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Step 2 — remove degenerate faces
# ---------------------------------------------------------------------------

def _remove_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    # Repeated vertex indices
    degenerate = (
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    )
    faces = faces[~degenerate]
    if len(faces) == 0:
        return faces

    # Zero-area faces
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross, axis=1)
    faces = faces[area > 1e-15]
    return faces


# ---------------------------------------------------------------------------
# Step 3 — remove unreferenced vertices and re-index
# ---------------------------------------------------------------------------

def _remove_unreferenced(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices, faces
    used = np.unique(faces)
    old2new = np.full(len(vertices), -1, dtype=np.int32)
    old2new[used] = np.arange(len(used), dtype=np.int32)
    new_verts = vertices[used]
    new_faces = old2new[faces]
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Step 4 — fix normals (BFS winding consistency)
# ---------------------------------------------------------------------------

def _fix_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Walk the face graph; flip winding of faces whose normal is inconsistent
    with the majority-orientation of already-visited neighbours.
    This is O(F) but requires edge adjacency.
    """
    if len(faces) == 0:
        return faces

    faces = faces.copy()

    # Build edge → list of face indices
    edge_to_faces: Dict = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for e in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted(e))].append(fi)

    face_normals = compute_face_normals(vertices, faces)
    visited = np.zeros(len(faces), dtype=bool)

    for seed in range(len(faces)):
        if visited[seed]:
            continue
        queue = deque([seed])
        visited[seed] = True
        while queue:
            fi = queue.popleft()
            a, b, c = faces[fi]
            for e in ((a, b), (b, c), (c, a)):
                for fj in edge_to_faces[tuple(sorted(e))]:
                    if visited[fj]:
                        continue
                    visited[fj] = True
                    # Flip fj if its normal opposes fi's normal
                    if np.dot(face_normals[fi], face_normals[fj]) < 0:
                        faces[fj, 1], faces[fj, 2] = faces[fj, 2], faces[fj, 1]
                        face_normals[fj] = -face_normals[fj]
                    queue.append(fj)

    return faces


# ---------------------------------------------------------------------------
# Step 5 — keep largest connected component
# ---------------------------------------------------------------------------

def _keep_largest_component(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Build vertex adjacency via union-find
    parent = np.arange(len(vertices), dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, c in faces:
        union(a, b)
        union(b, c)

    # Find largest component
    roots = np.array([find(i) for i in range(len(vertices))], dtype=np.int32)
    unique, counts = np.unique(roots, return_counts=True)
    largest_root = unique[counts.argmax()]

    keep_verts = np.where(roots == largest_root)[0]
    keep_mask = np.zeros(len(vertices), dtype=bool)
    keep_mask[keep_verts] = True

    face_mask = keep_mask[faces[:, 0]] & keep_mask[faces[:, 1]] & keep_mask[faces[:, 2]]
    return _remove_unreferenced(vertices, faces[face_mask])
