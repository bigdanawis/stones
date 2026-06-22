"""
Mesh decimation via Quadric Error Metrics (QEM).

  Garland & Heckbert 1997 — "Surface Simplification Using Quadric Error Metrics"

Two methods exposed:
  qem_decimate(V, F, target_faces)    — full QEM, best quality, O(F log F)
  cluster_decimate(V, F, target_faces)— vertex clustering, fast, lower quality

Hierarchical decimation:
  hierarchical_decimate(V, F, levels) — returns a list of (V, F) at LOD levels

Strategy for very large meshes (>200k faces):
  cluster first to ~200k, then run QEM for the remainder.
  Toggle with pre_cluster=True (default).
"""
import heapq
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    method: str = "qem",
    pre_cluster: bool = True,
    pre_cluster_limit: int = 150_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate a mesh to `target_faces` triangles.

    Parameters
    ----------
    vertices        : [V, 3] float64
    faces           : [F, 3] int32
    target_faces    : desired number of output faces
    method          : 'qem' | 'cluster'
    pre_cluster     : if True and F > pre_cluster_limit, run fast vertex
                      clustering first to reduce the mesh before QEM
    pre_cluster_limit : face count threshold that triggers pre-clustering
    """
    n_edges_in = _count_edges(faces)
    log.info(f"Decimate input : {len(vertices):>10,} V  {n_edges_in:>10,} E  {len(faces):>10,} F"
             f"  (method={method}, target={target_faces:,} F)")

    # Try trimesh first — it uses a compiled QEM that is 10–100x faster
    result = _try_trimesh_decimate(vertices, faces, target_faces)
    if result is not None:
        _log_decimate_result("trimesh-QEM", vertices, faces, result[0], result[1])
        return result

    if method == "cluster":
        result = cluster_decimate(vertices, faces, target_faces)
        _log_decimate_result("cluster", vertices, faces, result[0], result[1])
        return result

    # QEM — optionally pre-cluster on huge meshes
    if pre_cluster and len(faces) > pre_cluster_limit:
        intermediate = max(target_faces * 3, pre_cluster_limit)
        log.info(f"  Pre-clustering {len(faces):,} F -> {intermediate:,} F before QEM...")
        vertices, faces = cluster_decimate(vertices, faces, intermediate)
        log.info(f"  After pre-cluster: {len(vertices):,} V  {_count_edges(faces):,} E  {len(faces):,} F")

    result = qem_decimate(vertices, faces, target_faces)
    _log_decimate_result("QEM", vertices, faces, result[0], result[1])
    return result


def hierarchical_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    levels: int,
    min_faces: int = 500,
    method: str = "qem",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate `levels` LOD meshes from full resolution down to `min_faces`.

    Returns a list of (vertices, faces) tuples, index 0 = highest detail.
    The full-resolution mesh is NOT included (level 0 = first decimated level).
    """
    n = len(faces)
    if n <= min_faces or levels == 0:
        return [(vertices.copy(), faces.copy())]

    # Logarithmically spaced face counts: full -> min_faces
    counts = np.geomspace(n, min_faces, num=levels + 1, dtype=int)[1:]
    counts = np.unique(counts)[::-1]   # descending, deduplicated

    lods: List[Tuple[np.ndarray, np.ndarray]] = []
    v_cur, f_cur = vertices, faces

    for target in counts:
        if target >= len(f_cur):
            lods.append((v_cur.copy(), f_cur.copy()))
            continue
        log.info(f"LOD: {len(f_cur):,}F -> {target:,}F  ({method})")
        v_cur, f_cur = decimate(v_cur, f_cur, int(target), method=method)
        lods.append((v_cur.copy(), f_cur.copy()))
        if len(f_cur) <= min_faces:
            break

    return lods


# ---------------------------------------------------------------------------
# QEM decimation
# ---------------------------------------------------------------------------

def qem_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Quadric Error Metrics edge-collapse decimation.
    """
    if target_faces >= len(faces):
        return vertices.copy(), faces.copy()

    V = vertices.astype(np.float64)
    F = faces.astype(np.int32).copy()

    # -- 1. Compute per-vertex quadrics --
    quadrics = _compute_quadrics(V, F)

    # -- 2. Build vertex->face and vertex->edge adjacency --
    vert_faces: List[set] = [set() for _ in range(len(V))]
    for fi, (a, b, c) in enumerate(F):
        vert_faces[a].add(fi)
        vert_faces[b].add(fi)
        vert_faces[c].add(fi)

    # Unique edges: (u, v) with u < v
    edge_map: dict = {}    # (u,v) -> edge_id
    edges: List[Tuple[int, int]] = []
    vert_edges: List[set] = [set() for _ in range(len(V))]

    def _add_edge(u, v):
        key = (min(u, v), max(u, v))
        if key not in edge_map:
            ei = len(edges)
            edge_map[key] = ei
            edges.append(key)
            vert_edges[key[0]].add(ei)
            vert_edges[key[1]].add(ei)

    for a, b, c in F:
        _add_edge(a, b); _add_edge(b, c); _add_edge(a, c)

    log.info(f"  QEM setup: {len(V):,} V  {len(edges):,} E  {len(F):,} F"
             f"  -> target {target_faces:,} F  ({target_faces/len(F)*100:.1f}%)")

    # -- 3. Edge cost heap --
    edge_valid = bytearray(len(edges))  # 0 = valid, 1 = removed
    # We'll append new entries; initial capacity = len(edges)
    # Extend arrays lazily via lists

    heap: list = []
    optimal_pos: List[Optional[np.ndarray]] = [None] * len(edges)

    def _edge_cost_and_pos(u, v):
        Q = quadrics[u] + quadrics[v]
        A = Q[:3, :3]
        b = -Q[:3, 3]
        try:
            cond = np.linalg.cond(A)
            if cond < 1e10:
                x = np.linalg.solve(A, b)
            else:
                x = (V[u] + V[v]) * 0.5
        except np.linalg.LinAlgError:
            x = (V[u] + V[v]) * 0.5
        xh = np.append(x, 1.0)
        cost = max(0.0, float(xh @ Q @ xh))
        return cost, x

    for ei, (u, v) in enumerate(edges):
        cost, pos = _edge_cost_and_pos(u, v)
        optimal_pos[ei] = pos
        heapq.heappush(heap, (cost, ei, u, v))

    # -- 4. Union-find for vertex merging --
    parent = np.arange(len(V), dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    face_valid = np.ones(len(F), dtype=bool)
    current_faces = len(F)
    next_eid = [len(edges)]
    total_to_remove = current_faces - target_faces
    _next_log_at = [current_faces - max(1, total_to_remove // 10)]  # log every 10%

    # -- 5. Main collapse loop --
    while current_faces > target_faces and heap:
        cost, ei, u_orig, v_orig = heapq.heappop(heap)

        if ei >= len(edge_valid) or edge_valid[ei]:
            continue

        u, v = find(u_orig), find(v_orig)
        if u == v:
            edge_valid[ei] = 1
            continue

        # Always merge higher-index into lower
        if u > v:
            u, v = v, u

        # Move u to optimal position
        V[u] = optimal_pos[ei] if ei < len(optimal_pos) and optimal_pos[ei] is not None \
               else (V[u] + V[v]) * 0.5
        edge_valid[ei] = 1
        parent[v] = u
        quadrics[u] += quadrics[v]

        # Update faces that referenced v
        for fi in list(vert_faces[v]):
            if not face_valid[fi]:
                continue
            for k in range(3):
                if F[fi, k] == v:
                    F[fi, k] = u
            a, b, c = F[fi, 0], F[fi, 1], F[fi, 2]
            if a == b or b == c or a == c:
                face_valid[fi] = False
                current_faces -= 1
            else:
                vert_faces[u].add(fi)

        if current_faces <= _next_log_at[0]:
            pct = (1 - current_faces / len(F)) * 100
            log.info(f"  QEM progress: {current_faces:,} F remaining  ({pct:.0f}% removed)")
            _next_log_at[0] -= max(1, total_to_remove // 10)

        # Invalidate old edges incident to v; push updated costs for u
        for old_ei in list(vert_edges[v]):
            edge_valid[old_ei] = 1
            if old_ei >= len(edges):
                continue
            eu, ev = edges[old_ei]
            nu, nv = find(eu), find(ev)
            if nu == nv:
                continue
            if nu > nv:
                nu, nv = nv, nu
            new_ei = next_eid[0]
            next_eid[0] += 1
            edges.append((nu, nv))
            edge_valid.extend(b'\x00')
            c2, pos2 = _edge_cost_and_pos(nu, nv)
            optimal_pos.append(pos2)
            heapq.heappush(heap, (c2, new_ei, nu, nv))
            vert_edges[nu].add(new_ei)

    # -- 6. Rebuild clean mesh --
    # Redirect all vertices to their canonical root
    for fi in range(len(F)):
        if face_valid[fi]:
            F[fi, 0] = find(F[fi, 0])
            F[fi, 1] = find(F[fi, 1])
            F[fi, 2] = find(F[fi, 2])

    out_faces = F[face_valid]
    # Remove remaining degenerate faces
    degenerate = (out_faces[:, 0] == out_faces[:, 1]) | \
                 (out_faces[:, 1] == out_faces[:, 2]) | \
                 (out_faces[:, 0] == out_faces[:, 2])
    out_faces = out_faces[~degenerate]

    used = np.unique(out_faces)
    remap = np.full(len(V), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    out_verts = V[used]
    out_faces = remap[out_faces]

    n_edges_out = _count_edges(out_faces)
    log.info(f"  QEM result : {len(out_verts):>10,} V  {n_edges_out:>10,} E  {len(out_faces):>10,} F")
    return out_verts.astype(np.float64), out_faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Vertex clustering (fast, lower quality)
# ---------------------------------------------------------------------------

def cluster_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    grid_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition space into a uniform grid; merge all vertices in each cell.
    Fast (O(V)) but may destroy sharp features.
    Used as a pre-pass before QEM on huge meshes.
    """
    if target_faces >= len(faces):
        return vertices.copy(), faces.copy()

    V = vertices.astype(np.float64)
    ratio = target_faces / len(faces)

    if grid_size is None:
        # Heuristic: grid cells ≈ sqrt(ratio) * num_vertices^(1/3)
        grid_size = max(4, int(np.cbrt(len(V) * ratio) * 1.5))

    log.info(f"  Cluster: grid={grid_size}^3  {len(vertices):,} V  {_count_edges(faces):,} E  {len(faces):,} F  -> ~{target_faces:,} F")

    # Map each vertex to a grid cell
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    span = mx - mn
    span[span == 0] = 1.0

    cell = ((V - mn) / span * (grid_size - 1)).astype(np.int32)
    cell_id = cell[:, 0] * grid_size * grid_size + cell[:, 1] * grid_size + cell[:, 2]

    # Representative vertex per cell = mean position
    unique_cells, inv = np.unique(cell_id, return_inverse=True)
    new_V = np.zeros((len(unique_cells), 3), dtype=np.float64)
    counts = np.zeros(len(unique_cells), dtype=np.int32)
    np.add.at(new_V, inv, V)
    np.add.at(counts, inv, 1)
    new_V /= counts[:, None]

    # Remap faces
    new_F = inv[faces].astype(np.int32)
    degenerate = (new_F[:, 0] == new_F[:, 1]) | \
                 (new_F[:, 1] == new_F[:, 2]) | \
                 (new_F[:, 0] == new_F[:, 2])
    new_F = new_F[~degenerate]

    # Remove duplicated faces
    new_F_sorted = np.sort(new_F, axis=1)
    _, unique_fi = np.unique(new_F_sorted, axis=0, return_index=True)
    new_F = new_F[unique_fi]

    log.info(f"  Cluster result : {len(new_V):>10,} V  {_count_edges(new_F):>10,} E  {len(new_F):>10,} F")
    return new_V, new_F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_edges(faces: np.ndarray) -> int:
    """Count unique edges in a face array without building a full dict."""
    if len(faces) == 0:
        return 0
    edges = np.concatenate([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [0, 2]], axis=1),
    ], axis=0)
    return len(np.unique(edges, axis=0))


def _log_decimate_result(method: str, v_in, f_in, v_out, f_out):
    n_e_in  = _count_edges(f_in)
    n_e_out = _count_edges(f_out)
    pct_v = len(v_out) / max(len(v_in), 1) * 100
    pct_f = len(f_out) / max(len(f_in), 1) * 100
    log.info(
        f"Decimate output ({method}):\n"
        f"  vertices : {len(v_in):>10,} -> {len(v_out):>10,}  ({pct_v:.1f}% kept)\n"
        f"  edges    : {n_e_in:>10,} -> {n_e_out:>10,}  ({n_e_out/max(n_e_in,1)*100:.1f}% kept)\n"
        f"  faces    : {len(f_in):>10,} -> {len(f_out):>10,}  ({pct_f:.1f}% kept)"
    )


def _compute_quadrics(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return per-vertex 4×4 quadric matrices, shape [V, 4, 4]."""
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Face normals (unnormalised — area weighting comes for free)
    raw = np.cross(v1 - v0, v2 - v0)          # [F, 3]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    safe = norms[:, 0] > 0
    n = raw.copy()
    n[safe] /= norms[safe]

    d = -(n * v0).sum(axis=1, keepdims=True)   # [F, 1]
    p = np.concatenate([n, d], axis=1)          # [F, 4]  plane equation

    # Face quadric Kp = p p^T, shape [F, 4, 4]
    Kp = p[:, :, None] * p[:, None, :]         # [F, 4, 4]

    # Accumulate into per-vertex quadrics (area-weighted)
    area = norms[:, 0] * 0.5
    Q = np.zeros((len(V), 4, 4), dtype=np.float64)
    for i in range(3):
        np.add.at(Q, F[:, i], Kp * area[:, None, None])

    return Q


def _try_trimesh_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Use trimesh's compiled QEM if available (much faster on large meshes)."""
    try:
        import trimesh  # noqa: PLC0415
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        simplified = mesh.simplify_quadric_decimation(target_faces)
        if simplified is not None and len(simplified.faces) > 0:
            log.info(f"trimesh QEM: {len(faces):,}F -> {len(simplified.faces):,}F")
            return (
                np.array(simplified.vertices, dtype=np.float64),
                np.array(simplified.faces,    dtype=np.int32),
            )
    except Exception:
        pass
    return None
