"""
Surface point sampling with three modes:
  uniform     — area-weighted random sampling
  curvature   — bias toward high-curvature regions (ridges, scars)
  edge_aware  — 70% area-weighted + 30% high-dihedral-angle regions

Each mode returns points [N, 3] and normals [N, 3].
Designed for archaeological lithic tools where sharp edges and ridges matter.
"""
from typing import Tuple

import numpy as np

from src.mesh_cleaning import (
    compute_face_areas,
    compute_face_normals,
    compute_vertex_normals,
)
from src.utils import get_logger

log = get_logger(__name__)

EDGE_RATIO_DEFAULT = 0.30   # fraction of points from high-dihedral zones


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sample_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    mode: str = "uniform",
    edge_ratio: float = EDGE_RATIO_DEFAULT,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample `num_points` surface points from a triangle mesh.

    Parameters
    ----------
    vertices   : [V, 3]
    faces      : [F, 3]
    num_points : target number of output points
    mode       : 'uniform' | 'curvature' | 'edge_aware'
    edge_ratio : fraction used for edge zone in edge_aware mode
    rng        : optional numpy random Generator for reproducibility

    Returns
    -------
    points  : [N, 3]  float32
    normals : [N, 3]  float32
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(faces) == 0:
        raise ValueError("Mesh has no faces after cleaning.")

    if mode == "uniform":
        pts, nrm = _sample_uniform(vertices, faces, num_points, rng)
    elif mode == "curvature":
        pts, nrm = _sample_curvature(vertices, faces, num_points, rng)
    elif mode == "edge_aware":
        pts, nrm = _sample_edge_aware(vertices, faces, num_points, edge_ratio, rng)
    else:
        raise ValueError(f"Unknown sampling mode: {mode!r}")

    # Guarantee exactly num_points (repeat-sample if mesh is tiny)
    if len(pts) < num_points:
        idx = rng.integers(0, len(pts), size=num_points - len(pts))
        pts = np.concatenate([pts, pts[idx]], axis=0)
        nrm = np.concatenate([nrm, nrm[idx]], axis=0)
    elif len(pts) > num_points:
        idx = rng.choice(len(pts), num_points, replace=False)
        pts, nrm = pts[idx], nrm[idx]

    return pts.astype(np.float32), nrm.astype(np.float32)


# ---------------------------------------------------------------------------
# Mode A — uniform area-weighted sampling
# ---------------------------------------------------------------------------

def _sample_uniform(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    areas = compute_face_areas(vertices, faces)
    weights = areas / areas.sum()
    return _sample_from_weights(vertices, faces, weights, num_points, rng)


# ---------------------------------------------------------------------------
# Mode B — curvature-weighted sampling
# ---------------------------------------------------------------------------

def _sample_curvature(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vertex curvature estimated as average angular deviation of adjacent face
    normals from the vertex normal (angular variation proxy).
    High curvature → ridges, scars, bulge-tips.
    """
    vn = compute_vertex_normals(vertices, faces)    # [V, 3]
    fn = compute_face_normals(vertices, faces)       # [F, 3]
    areas = compute_face_areas(vertices, faces)      # [F]

    # For each face, mean curvature weight = average vertex curvature of its corners
    vert_curv = _estimate_vertex_curvature(vertices, faces, vn, fn)

    face_curv = vert_curv[faces].mean(axis=1)       # [F]
    face_curv = np.clip(face_curv, 0, None)

    # Combine area and curvature: w = area * (1 + alpha * curvature)
    alpha = 3.0
    weights = areas * (1.0 + alpha * face_curv)
    weights /= weights.sum()

    return _sample_from_weights(vertices, faces, weights, num_points, rng)


def _estimate_vertex_curvature(
    vertices: np.ndarray,
    faces: np.ndarray,
    vn: np.ndarray,
    fn: np.ndarray,
) -> np.ndarray:
    """
    Per-vertex curvature as mean angular difference (radians) between the
    vertex normal and its adjacent face normals. Range [0, π].
    """
    curv = np.zeros(len(vertices), dtype=np.float64)
    count = np.zeros(len(vertices), dtype=np.int32)

    for fi in range(len(faces)):
        for vi in faces[fi]:
            dot = np.clip(np.dot(vn[vi], fn[fi]), -1.0, 1.0)
            curv[vi] += np.arccos(dot)
            count[vi] += 1

    nonzero = count > 0
    curv[nonzero] /= count[nonzero]
    return curv


# ---------------------------------------------------------------------------
# Mode C — edge-aware sampling
# ---------------------------------------------------------------------------

def _sample_edge_aware(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    edge_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    70% points from area-weighted sampling,
    30% from faces adjacent to high-dihedral edges (sharp ridges / scars).
    Ratio is controlled by edge_ratio.
    """
    n_edge = max(1, int(num_points * edge_ratio))
    n_surf = num_points - n_edge

    # --- surface portion ---
    areas = compute_face_areas(vertices, faces)
    w_surf = areas / areas.sum()
    pts_s, nrm_s = _sample_from_weights(vertices, faces, w_surf, n_surf, rng)

    # --- edge / dihedral portion ---
    dihedral_face_weights = _compute_dihedral_face_weights(vertices, faces, areas)
    pts_e, nrm_e = _sample_from_weights(vertices, faces, dihedral_face_weights, n_edge, rng)

    pts = np.concatenate([pts_s, pts_e], axis=0)
    nrm = np.concatenate([nrm_s, nrm_e], axis=0)
    return pts, nrm


def _compute_dihedral_face_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """
    For each face, its weight is proportional to the maximum dihedral angle
    across all its edges (larger dihedral = sharper edge = more important).
    Faces not adjacent to any sharp edge get weight = area only.
    """
    fn = compute_face_normals(vertices, faces)          # [F, 3]

    # Build edge → list of (face_index, edge_position) mapping
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for e in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted(e))].append(fi)

    # Per-face maximum dihedral angle (radians, 0 = planar, π = sharp crease)
    max_dihedral = np.zeros(len(faces), dtype=np.float64)
    for adj_faces in edge_faces.values():
        if len(adj_faces) != 2:
            continue
        fi, fj = adj_faces
        dot = np.clip(np.dot(fn[fi], fn[fj]), -1.0, 1.0)
        angle = np.arccos(-dot)   # dihedral = π - angle between outward normals
        max_dihedral[fi] = max(max_dihedral[fi], angle)
        max_dihedral[fj] = max(max_dihedral[fj], angle)

    # Weight: area * (1 + beta * dihedral / π)
    beta = 5.0
    weights = areas * (1.0 + beta * max_dihedral / np.pi)
    weights /= weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Core sampler: random point on mesh surface
# ---------------------------------------------------------------------------

def _sample_from_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly within triangles selected by `weights`.
    Returns points [N, 3] and per-point normals [N, 3].
    """
    # Sample face indices according to weights
    face_idx = rng.choice(len(faces), size=num_points, p=weights)

    v0 = vertices[faces[face_idx, 0]]
    v1 = vertices[faces[face_idx, 1]]
    v2 = vertices[faces[face_idx, 2]]

    # Random barycentric coordinates (uniform over triangle)
    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    pts = (u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2)

    # Face normals as point normals
    fn = compute_face_normals(vertices, faces)
    nrm = fn[face_idx]

    return pts, nrm
