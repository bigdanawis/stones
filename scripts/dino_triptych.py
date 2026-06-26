"""
DINOv2 embedding via 6-view orthographic depth maps.

After SVD alignment of each mesh (X=longest, Y=intermediate, Z=shortest/thickness),
six orthographic depth images are rendered — one per axis direction:

  pZ / nZ  — top / bottom  (plan view,    h=X, v=Y)  encodes plan shape
  pX / nX  — right / left  (side profile, h=Y, v=Z)  encodes length × thickness
  pY / nY  — front / back  (end profile,  h=X, v=Z)  encodes width  × thickness

Each image: for every pixel the depth of the nearest surface along the view
axis, normalised by the global per-axis maximum across all stones.
Background (no mesh) = 0.

Each view is fed independently to standard 3-channel DINOv2 (grayscale broadcast
to RGB); the 6 CLS tokens are mean-pooled into a single embedding.
No architecture changes required.

Renders saved to  outputs/multiview_renders/.

Usage
-----
  python scripts/dino_triptych.py
  python scripts/dino_triptych.py --skip_clean --decimate --target_ratio 0.10
  python scripts/dino_triptych.py --pca_only
  python scripts/dino_triptych.py --pca_only --run_dir outputs/dino_triptych/20260625_120000
  python scripts/dino_triptych.py --reload_scale outputs/multiview_scale.json
  python scripts/dino_triptych.py --finetune --ft_blocks 2 --ft_epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import rotate as nd_rotate, binary_erosion
from scipy.spatial  import ConvexHull
from scipy.stats    import pearsonr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_cleaning import clean_mesh, compute_face_normals
from src.mesh_io import load_mesh
from src.utils import collect_mesh_files, ensure_dir, get_logger
from scripts.demo_embed_pca import (
    load_site_map,
    plot_similarity_matrix,
    _PALETTE,
    _scale_and_reduce,
    _site_colors,
    _plotly_3d,
)

log = get_logger("dino_triptych")

IMAGE_SIZE = 224
FILL_FRAC  = 0.85
DINO_MODEL = "dinov2_vits14"

# 6 orthographic views: (name, h_axis, v_axis, d_axis, camera_at_positive_inf)
# After SVD alignment: axis 0=X=longest, 1=Y=intermediate, 2=Z=shortest
VIEWS = [
    ("pZ", 0, 1, 2, True),   # top    — looking down   (plan view)
    ("nZ", 0, 1, 2, False),  # bottom — looking up
    ("pX", 1, 2, 0, True),   # right  — looking left   (side profile)
    ("nX", 1, 2, 0, False),  # left   — looking right
    ("pY", 0, 2, 1, True),   # front  — looking back   (front profile)
    ("nY", 0, 2, 1, False),  # back   — looking front
]

WRL_DIR      = ROOT / "wrl"
OUTPUT_BASE  = ROOT / "outputs" / "dino_triptych"
SCALE_DIR    = ROOT / "outputs"
RENDERS_DIR  = ROOT / "outputs" / "multiview_renders"
SCALE_FILE   = SCALE_DIR / "multiview_scale.json"
XLSX_DEFAULT = WRL_DIR / "Handaxes 2026 list with sites.xlsx"


# ---------------------------------------------------------------------------
# Pass 1 — global per-axis depth ranges
# ---------------------------------------------------------------------------

def find_global_params(files: list[Path]) -> tuple[float, float, float]:
    """
    Scan all meshes and return the maximum sorted extents
    (max_long, max_mid, max_short) across the dataset.

    Extents are sorted largest→smallest per mesh so they correspond to
    SVD-aligned axes X, Y, Z regardless of original file orientation.
    """
    log.info(f"Pass 1: scanning {len(files)} meshes for global extents …")
    max_x = max_y = max_z = 0.0
    for p in tqdm(files, unit="mesh", desc="Pass 1"):
        try:
            v   = load_mesh(p)["vertices"]
            ext = np.sort(v.max(axis=0) - v.min(axis=0))[::-1]
            max_x = max(max_x, float(ext[0]))
            max_y = max(max_y, float(ext[1]))
            max_z = max(max_z, float(ext[2]))
        except Exception:
            pass
    max_x, max_y, max_z = max(max_x, 1e-10), max(max_y, 1e-10), max(max_z, 1e-10)
    log.info(f"  max_long={max_x:.4f}  max_mid={max_y:.4f}  max_short={max_z:.4f}")
    return max_x, max_y, max_z


def save_scale(max_x: float, max_y: float, max_z: float,
               path: Path = SCALE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"max_long": max_x, "max_mid": max_y, "max_short": max_z}, fh, indent=2)
    log.info(f"Scale saved → {path}")


def load_scale(path: Path = SCALE_FILE) -> tuple[float, float, float]:
    with open(path) as fh:
        d = json.load(fh)
    mx, my, mz = float(d["max_long"]), float(d["max_mid"]), float(d["max_short"])
    log.info(f"Scale loaded: max_long={mx:.4f}  max_mid={my:.4f}  max_short={mz:.4f}")
    return mx, my, mz


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _poly_render(polys: np.ndarray, colors: np.ndarray, H: int, W: int) -> np.ndarray:
    """Rasterise triangles onto a black canvas. Returns [H, W, 3] uint8."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor((0, 0, 0))
    fig.patch.set_facecolor((0, 0, 0))
    ax.add_collection(
        PolyCollection(polys, facecolors=colors, edgecolors="none", antialiased=False)
    )
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    if h != H or w != W:
        buf = np.array(Image.fromarray(buf).resize((W, H), Image.BILINEAR))
    return buf


def _depth_view(
    vertices:     np.ndarray,   # [V, 3] SVD-aligned, centred
    faces:        np.ndarray,   # [F, 3]
    h_axis:       int,          # world axis → horizontal pixel
    v_axis:       int,          # world axis → vertical pixel
    d_axis:       int,          # world axis → depth
    positive:     bool,         # True = camera at +∞ (nearest = highest d)
    canvas:       int,
    global_max_d: float,
) -> np.ndarray:                # [canvas, canvas] float32 in [0, 1]
    """
    Render one orthographic depth image.

    positive=True  : camera at +∞ along d_axis; faces sorted ascending so the
                     highest-d surface (nearest to camera) overwrites lower ones.
    positive=False : camera at -∞; faces sorted descending so the lowest-d
                     surface (nearest) overwrites.

    Depth value is normalised so the nearest possible surface = 1, background = 0.
    A small sentinel (1/255) ensures stone pixels are always > 0.
    """
    vh = vertices[:, h_axis]
    vv = vertices[:, v_axis]
    vd = vertices[:, d_axis]

    scale = canvas * FILL_FRAC / max(float(vh.max() - vh.min()),
                                     float(vv.max() - vv.min()), 1e-10)
    ctr_h = (vh.min() + vh.max()) * 0.5
    ctr_v = (vv.min() + vv.max()) * 0.5

    ph    = (vh - ctr_h) * scale + canvas * 0.5
    pv    = (vv - ctr_v) * scale + canvas * 0.5
    v2d   = np.stack([ph, pv], axis=1)
    tris  = v2d[faces]
    avg_d = vd[faces].mean(axis=1)

    d_min  = float(vd.min())
    d_max  = float(vd.max())
    norm_d = max(global_max_d, d_max - d_min, 1e-10)
    S      = 1.0 / 255.0   # sentinel so stone base > 0, background stays 0

    if positive:
        order      = np.argsort(avg_d)        # low first → high d wins
        depth_vals = avg_d - d_min
    else:
        order      = np.argsort(-avg_d)       # high first → low d wins
        depth_vals = d_max - avg_d

    enc = S + (1.0 - S) * np.clip(depth_vals[order] / norm_d, 0.0, 1.0)
    rgb = _poly_render(tris[order], np.stack([enc, enc, enc], axis=1), canvas, canvas)
    return rgb[:, :, 0].astype(np.float32) / 255.0


def render_multiview(
    vertices:     np.ndarray,
    faces:        np.ndarray,
    global_max_x: float,
    global_max_y: float,
    global_max_z: float,
    canvas:       int = IMAGE_SIZE,
) -> list[np.ndarray]:          # 6 × [canvas, canvas] float32
    """
    SVD-align the mesh then render 6 orthographic depth images in VIEWS order.
    Returns a list of 6 grayscale [canvas, canvas] float32 arrays.
    """
    centred  = vertices - vertices.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    verts    = centred @ Vt.T   # X=longest, Y=intermediate, Z=shortest

    max_per_axis = [global_max_x, global_max_y, global_max_z]

    return [
        _depth_view(verts, faces, h, v, d, pos, canvas, max_per_axis[d])
        for _, h, v, d, pos in VIEWS
    ]


def save_multiview(views: list[np.ndarray], stem: str, rd: Path) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    for img, (name, *_) in zip(views, VIEWS):
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(rd / f"{stem}_{name}.png")


def load_multiview(stem: str, rd: Path) -> list[np.ndarray]:
    return [
        np.array(Image.open(rd / f"{stem}_{name}.png"), dtype=np.float32) / 255.0
        for name, *_ in VIEWS
    ]


# ---------------------------------------------------------------------------
# DINOv2 — standard 3-channel, mean-pool over 6 views
# ---------------------------------------------------------------------------

def build_dino(model_name: str = DINO_MODEL, device=None):
    log.info(f"Loading DINOv2 {model_name} (standard 3-channel, 6-view pool) …")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval()
    return model.to(device) if device is not None else model


def _view_to_tensor(img: np.ndarray, device: torch.device,
                    img_size: int) -> torch.Tensor:
    """[H, W] grayscale → [1, 3, img_size, img_size] tensor on device, [-1, 1]."""
    H, W = img.shape
    sq   = max(H, W)
    if H != W:
        padded = np.zeros((sq, sq), dtype=np.float32)
        padded[(sq - H) // 2:(sq - H) // 2 + H,
               (sq - W) // 2:(sq - W) // 2 + W] = img
        img = padded
    if sq != img_size:
        img = np.array(
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            .resize((img_size, img_size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    t = torch.from_numpy(img[np.newaxis]).repeat(3, 1, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0   # [1, 3, H, W]


@torch.no_grad()
def embed_multiview(
    model,
    views:    list[np.ndarray],
    device:   torch.device,
    img_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Run DINOv2 on all 6 grayscale depth views in a single batched forward pass
    and mean-pool the CLS tokens.  Returns [D] float32.
    """
    batch = torch.cat([_view_to_tensor(v, device, img_size) for v in views], dim=0)  # [6, 3, H, W]
    cls   = model(batch)                  # [6, D]
    return cls.mean(dim=0).cpu().numpy().astype(np.float32)   # [D]


# ---------------------------------------------------------------------------
# Optional SimCLR finetuning
# ---------------------------------------------------------------------------

def _augment(views: list[np.ndarray]) -> list[np.ndarray]:
    """Independent random rotation per view (scalar depth maps need no direction fix)."""
    return [
        np.clip(
            nd_rotate(v, angle=np.random.uniform(0.0, 360.0),
                      reshape=False, mode="constant", cval=0.0),
            0.0, 1.0,
        ).astype(np.float32)
        for v in views
    ]


def _views_to_batch(views: list[np.ndarray], device: torch.device,
                    img_size: int) -> torch.Tensor:
    """6-view list of [H, W] → [6, 3, img_size, img_size] tensor, [-1, 1]."""
    return torch.cat([_view_to_tensor(v, device, img_size) for v in views], dim=0)


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class _NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T
        sim.masked_fill_(torch.eye(2 * N, dtype=torch.bool, device=z.device), float("-inf"))
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
        return F.cross_entropy(sim, labels)


def _set_trainable(model: nn.Module, n_unfreeze_blocks: int) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    if n_unfreeze_blocks == -1:
        for p in model.parameters():
            p.requires_grad_(True)
        return
    if n_unfreeze_blocks == 0:
        return   # backbone fully frozen; only projection head trains
    for p in model.patch_embed.parameters():
        p.requires_grad_(True)
    for blk in list(model.blocks)[-n_unfreeze_blocks:]:
        for p in blk.parameters():
            p.requires_grad_(True)


def finetune_simclr(
    model:             nn.Module,
    images:            list[list[np.ndarray]],  # N × 6 × [H, W]
    device:            torch.device,
    n_epochs:          int   = 10,
    lr:                float = 1e-4,
    batch_size:        int   = 16,
    n_unfreeze_blocks: int   = 2,
    temperature:       float = 0.07,
    img_size:          int   = IMAGE_SIZE,
) -> nn.Module:
    """
    SimCLR finetuning on the 6-view multi-view representation.

    Positive pair: two independently augmented (rotated) sets of 6 views for
    the same stone.  Each set is embedded by running DINOv2 on all 6 views in
    one batched forward pass and mean-pooling the CLS tokens.
    """
    _set_trainable(model, n_unfreeze_blocks)
    model.train()

    head      = _ProjectionHead(in_dim=model.embed_dim).to(device)
    criterion = _NTXentLoss(temperature)
    params    = (list(filter(lambda p: p.requires_grad, model.parameters()))
                 + list(head.parameters()))
    opt       = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    n = len(images)
    log.info(
        f"SimCLR finetune: {n} stones  epochs={n_epochs}  batch={batch_size}  "
        f"lr={lr}  unfreeze_blocks={n_unfreeze_blocks}  T={temperature}"
    )

    for epoch in range(1, n_epochs + 1):
        idx        = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch = idx[start: start + batch_size]
            if len(batch) < 2:
                continue

            B = len(batch)
            # Stack all 6 augmented views for the whole batch → [B*6, 3, H, W]
            t1 = torch.cat([_views_to_batch(_augment(images[i]), device, img_size)
                             for i in batch])
            t2 = torch.cat([_views_to_batch(_augment(images[i]), device, img_size)
                             for i in batch])

            # Forward → [B*6, D] → reshape → [B, 6, D] → mean-pool → [B, D]
            e1 = model(t1).reshape(B, len(VIEWS), -1).mean(dim=1)
            e2 = model(t2).reshape(B, len(VIEWS), -1).mean(dim=1)

            loss = criterion(head(e1), head(e2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        log.info(f"  epoch {epoch:>3}/{n_epochs}  loss={epoch_loss / max(n_batches, 1):.4f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_SITE_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p", "<", ">"]


def _plot_2d(stems, coords, site_map, notes_map, title, xlabel, ylabel, out_path):
    stem_sites, unique_sites, site_color = _site_colors(stems, site_map)
    site_marker = {s: _SITE_MARKERS[i % len(_SITE_MARKERS)] for i, s in enumerate(unique_sites)}
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
        for i, (x, y) in enumerate(coords[:, :2]):
            ax.scatter(x, y, color=_PALETTE[i % len(_PALETTE)], s=120, zorder=3,
                       edgecolors="white", linewidths=0.5)
    if notes_map:
        import matplotlib.patheffects as pe
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


def plot_pca(stems, embeddings, out, site_map=None, notes_map=None):
    coords, var, _ = _scale_and_reduce(embeddings, n_components=10)
    _plot_2d(stems, coords, site_map, notes_map,
             f"6-view DINOv2 — PCA  ({len(stems)} stones)",
             f"PC1  ({var[0]:.1%} var)", f"PC2  ({var[1]:.1%} var)",
             out / "pca.png")


def plot_pca_13(stems, embeddings, out, site_map=None, notes_map=None):
    coords, var, _ = _scale_and_reduce(embeddings, n_components=10)
    coords13 = np.stack([coords[:, 0], coords[:, 2]], axis=1)
    _plot_2d(stems, coords13, site_map, notes_map,
             f"6-view DINOv2 — PC1 vs PC3  ({len(stems)} stones)",
             f"PC1  ({var[0]:.1%} var)", f"PC3  ({var[2]:.1%} var)",
             out / "pca_13.png")


def _fit_archetypes(X: np.ndarray, K: int = 4, n_iter: int = 500,
                    n_restarts: int = 3, seed: int = 0) -> np.ndarray:
    """
    Pareto Archetypal Analysis via Frank-Wolfe (Mørup & Hansen 2012 / Uri Alon).

    Finds K archetype points as convex combinations of data points such that
    every stone is approximately a convex combination of the archetypes:
        X ≈ S @ Z,   Z = C @ X
    where S (N×K) and C (K×N) have non-negative rows that each sum to 1.

    Frank-Wolfe updates are used instead of projected gradient: each step moves
    toward the vertex of the simplex that minimises the linearised objective,
    so no learning-rate tuning is needed and the simplex constraint is always
    satisfied exactly.  Data is column-normalised before fitting so the result
    is independent of PCA axis scale.
    """
    X = X.astype(np.float64)
    N, D = X.shape

    # Normalise each column to unit std — keeps gradient magnitudes O(1)
    # regardless of how much variance each PC carries.
    col_std = X.std(axis=0).clip(1e-8)
    Xn = X / col_std   # work in normalised space; results are transformed back

    def _run(rng: np.random.Generator) -> tuple[np.ndarray, float]:
        # ── Furthest-point init: spread initial archetypes across the cloud ──
        chosen = [int(rng.integers(N))]
        for _ in range(K - 1):
            dists = np.min(
                [np.sum((Xn - Xn[c]) ** 2, axis=1) for c in chosen], axis=0
            )
            chosen.append(int(rng.choice(N, p=dists / (dists.sum() + 1e-12))))

        C = np.zeros((K, N))            # K×N  archetype ← data weights
        for k, c in enumerate(chosen):
            C[k, c] = 1.0
        S = np.ones((N, K)) / K         # N×K  stone ← archetype weights

        for t in range(n_iter):
            # Frank-Wolfe step size: γ = 2/(t+2), decreases to 0
            gamma = 2.0 / (t + 2)

            Z = C @ Xn                  # K×D  current archetype positions
            R = Xn - S @ Z             # N×D  residuals

            # ── Update S: each stone finds its closest archetype ──────────
            # gS[i,k] = ∂L/∂S[i,k] = −2·R[i]·Z[k]  →  shape N×K
            gS = -R @ Z.T              # N×K; Z.T is D×K
            k_stars = np.argmin(gS, axis=1)         # N, — one argmin per stone
            S_fw = np.zeros((N, K))
            S_fw[np.arange(N), k_stars] = 1.0
            S = (1 - gamma) * S + gamma * S_fw

            # ── Update C: each archetype moves toward one data point ──────
            R = Xn - S @ Z             # recompute after S step
            # gC[k,n] = ∂L/∂C[k,n] = Σ_d (∂L/∂Z)[k,d]·Xn[n,d]
            #         = (−S^T R @ Xn^T)[k,n],  shape K×N
            gC = -(S.T @ R) @ Xn.T    # (K,D)@(D,N) = K×N
            n_stars = np.argmin(gC, axis=1)         # K, — one argmin per archetype
            C_fw = np.zeros((K, N))
            C_fw[np.arange(K), n_stars] = 1.0
            C = (1 - gamma) * C + gamma * C_fw

        Z_norm = C @ Xn
        loss   = float(np.mean((Xn - S @ Z_norm) ** 2))
        return C @ X, S, loss          # archetypes in original scale; S is (N,K)

    best_Z, best_S, best_loss = None, None, np.inf
    for r in range(n_restarts):
        Z, S_final, loss = _run(np.random.default_rng(seed + r))
        if loss < best_loss:
            best_Z, best_S, best_loss = Z, S_final, loss
    log.info(f"  archetypes  K={K}  loss={best_loss:.5f}")
    return best_Z, best_S   # (K, D),  (N, K)


def _expand_archetypes(Z: np.ndarray, factor: float = 0.20) -> np.ndarray:
    """Push each archetype outward from the centroid so more points fall inside."""
    centroid = Z.mean(axis=0)
    return centroid + (1.0 + factor) * (Z - centroid)


def _inside_tetrahedron(X: np.ndarray, Z: np.ndarray) -> int:
    """Count how many rows of X (N×3) lie inside the tetrahedron with vertices Z (4×3)."""
    T = (Z[1:] - Z[0]).T          # 3×3 edge matrix
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return 0
    bary = (X - Z[0]) @ T_inv.T   # N×3  barycentric coords for faces 1-3
    l4   = 1.0 - bary.sum(axis=1) # N    barycentric coord for face 0
    return int(((bary >= -1e-9).all(axis=1) & (l4 >= -1e-9)).sum())


def _pz_stats(stems: list[str], renders_dir: Path) -> tuple[list[str], np.ndarray]:
    """
    Compute per-stone statistics from the multiview depth renders.

    Shape statistics (from pZ silhouette mask):
      n_pixels, aspect_ratio, circularity, solidity, tip_sharpness

    Depth-mean per view (proxy for thickness in each projection direction):
      depth_pZ, depth_nZ, depth_pX, depth_nX, depth_pY, depth_nY

    Returns (stat_names, S_matrix) where S_matrix is (N, n_stats) with NaN
    for missing values.
    """
    VIEWS_ORDER = ["pZ", "nZ", "pX", "nX", "pY", "nY"]
    MASK_THRESH = 1.5 / 255.0
    PIX_THRESH  = 0.5 / 255.0

    stat_names = (["n_pixels", "aspect_ratio", "circularity", "solidity", "tip_sharpness"] +
                  [f"depth_{v}" for v in VIEWS_ORDER])
    n_stats    = len(stat_names)
    rows       = []

    for stem in stems:
        row = [np.nan] * n_stats

        # ── depth mean per view ───────────────────────────────────────────
        for vi, vname in enumerate(VIEWS_ORDER):
            p = renders_dir / f"{stem}_{vname}.png"
            if p.exists():
                img = np.array(Image.open(p), dtype=np.float32) / 255.0
                if img.ndim == 3:
                    img = img.mean(axis=2)
                vals = img[img > PIX_THRESH]
                if len(vals):
                    row[5 + vi] = float(vals.mean())

        # ── shape stats from pZ silhouette ───────────────────────────────
        pz = renders_dir / f"{stem}_pZ.png"
        if pz.exists():
            img  = np.array(Image.open(pz), dtype=np.float32) / 255.0
            if img.ndim == 3:
                img = img.mean(axis=2)
            mask = img > MASK_THRESH
            ys, xs = np.where(mask)
            n_pix  = len(ys)
            if n_pix >= 20:
                row[0] = float(n_pix)

                # aspect ratio via PCA on pixel coords
                coords_c = np.stack([xs, ys], axis=1).astype(np.float64)
                coords_c = coords_c - coords_c.mean(axis=0)
                cov      = (coords_c.T @ coords_c) / n_pix
                eigvals  = np.linalg.eigvalsh(cov)[::-1]
                row[1]   = float(np.sqrt(eigvals[0] / max(eigvals[1], 1e-8)))

                # circularity
                boundary   = mask & ~binary_erosion(mask)
                perimeter  = float(boundary.sum())
                row[2]     = float(4 * np.pi * n_pix / max(perimeter ** 2, 1.0))

                # solidity
                try:
                    hull   = ConvexHull(np.stack([xs, ys], axis=1).astype(np.float64))
                    row[3] = float(n_pix / max(hull.volume, 1.0))
                except Exception:
                    pass

                # tip sharpness
                eigvecs  = np.linalg.eigh(cov)[1][:, ::-1]
                lp       = coords_c @ eigvecs[:, 0]
                pp       = coords_c @ eigvecs[:, 1]
                p_range  = lp.max() - lp.min()
                tip_frac = 0.10
                def _tw(sel):
                    p = pp[sel]
                    return float(p.max() - p.min()) if sel.sum() >= 3 else np.nan
                tw1 = _tw(lp <= lp.min() + p_range * tip_frac)
                tw2 = _tw(lp >= lp.max() - p_range * tip_frac)
                if not (np.isnan(tw1) or np.isnan(tw2)):
                    row[4] = float(min(tw1, tw2) / max(pp.max() - pp.min(), 1e-8))

        rows.append(row)

    return stat_names, np.array(rows, dtype=np.float64)   # (N, n_stats)


def _archetype_report_figure(stat_names, vertex_labels, edge_labels,
                              r_vertex, p_vertex, r_edge, p_edge, out_png: Path):
    """Two-panel heatmap: vertex profiles (left) and edge-walk gradients (right).
    Significant cells (p<0.05) are fully opaque; non-significant ones are faded."""
    n_stats  = len(stat_names)
    K        = len(vertex_labels)
    n_edges  = len(edge_labels)
    VMAX     = 0.85
    cmap     = plt.cm.RdBu_r   # red = positive, blue = negative

    fig, axes = plt.subplots(
        1, 2,
        figsize=(3.5 + K * 1.3 + n_edges * 1.1, 1.5 + n_stats * 0.62),
        facecolor="#0f0f1a",
        gridspec_kw={"width_ratios": [K, n_edges], "wspace": 0.06},
    )

    def _draw(ax, R, P, col_labels, title):
        n_r, n_c = R.shape
        for si in range(n_r):
            for ci in range(n_c):
                r = R[si, ci]
                p = P[si, ci]
                if np.isnan(r):
                    continue
                sig   = p < 0.05
                norm_r = np.clip((r / VMAX + 1) / 2, 0, 1)
                color  = cmap(norm_r)
                # faint cell for non-significant, strong for significant
                rect = plt.Rectangle(
                    [ci - 0.5, si - 0.5], 1, 1,
                    facecolor=(*color[:3], 0.82 if sig else 0.18),
                    linewidth=0,
                )
                ax.add_patch(rect)
                stars = ("***" if p < 0.001 else "**" if p < 0.01
                         else "*" if p < 0.05 else "")
                ax.text(
                    ci, si, f"{r:+.2f}{stars}",
                    ha="center", va="center", fontsize=7,
                    color="white", alpha=1.0 if sig else 0.30,
                    fontweight="bold" if sig else "normal",
                )

        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(n_r - 0.5, -0.5)   # first stat at top
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(col_labels, fontsize=9, color="#ddd",
                           rotation=35, ha="right")
        ax.set_yticks(range(n_r))
        ax.set_facecolor("#12122a")
        ax.set_title(title, color="#eee", fontsize=10, pad=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.tick_params(colors="#555", length=0)
        for x in np.arange(-0.5, n_c, 1):
            ax.axvline(x, color="#1e1e3a", lw=0.6)
        for y in np.arange(-0.5, n_r, 1):
            ax.axhline(y, color="#1e1e3a", lw=0.6)

    _draw(axes[0], r_vertex, p_vertex, vertex_labels,
          "Vertex Profiles  (S[:,k] vs stat)")
    axes[0].set_yticklabels(stat_names, fontsize=8, color="#ccc")

    _draw(axes[1], r_edge, p_edge, edge_labels,
          "Edge-Walk Gradients  (position vs stat)")
    axes[1].set_yticklabels([])   # share y-axis labels with left panel

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-VMAX, VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, aspect=22, pad=0.01)
    cbar.set_label("Pearson r", color="#ccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#777")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccc", fontsize=8)

    fig.suptitle("Archetype Correlation Report  (faded = p ≥ 0.05)",
                 color="#ddd", fontsize=11, y=1.01)
    fig.savefig(out_png, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  -> {out_png}")


def _archetype_report(stems, coords3, S, Z, stat_names, stat_mat, out_txt: Path):
    """
    Print and save two correlation tables plus a visual heatmap figure:

    (A) Vertex profiles — Pearson r between S[:,k] (mixture weight for archetype k)
        and each image statistic.  High r means that property is elevated near Ak.

    (B) Edge-walk gradients — for each of the 6 tetrahedron edges Ak→Al, project
        each stone onto the edge direction and correlate that position with each stat.
        Tells you what changes as you walk from one archetype to another.

    Saves archetype_report.txt and archetype_report.png in the same directory.
    """
    N, K     = S.shape
    n_stats  = len(stat_names)
    edges    = [(i, j) for i in range(K) for j in range(i + 1, K)]
    n_edges  = len(edges)
    STARS    = lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    # ── pre-compute all Pearson r / p ─────────────────────────────────────────
    r_vertex = np.full((n_stats, K),       np.nan)
    p_vertex = np.full((n_stats, K),       np.nan)
    r_edge   = np.full((n_stats, n_edges), np.nan)
    p_edge   = np.full((n_stats, n_edges), np.nan)

    # pre-compute edge projections once
    t_edges = []
    for ki, kj in edges:
        ev    = Z[kj] - Z[ki]
        norm2 = ev @ ev
        t_edges.append((coords3 - Z[ki]) @ ev / norm2 if norm2 > 1e-12 else None)

    for si in range(n_stats):
        col   = stat_mat[:, si]
        valid = ~np.isnan(col)
        if valid.sum() < 3:
            continue
        for k in range(K):
            r, p = pearsonr(S[valid, k], col[valid])
            r_vertex[si, k] = r
            p_vertex[si, k] = p
        for ei, t in enumerate(t_edges):
            if t is not None:
                r, p = pearsonr(t[valid], col[valid])
                r_edge[si, ei] = r
                p_edge[si, ei] = p

    vertex_labels = [f"A{k+1}" for k in range(K)]
    edge_labels   = [f"A{i+1}→A{j+1}" for i, j in edges]

    lines = []
    def pr(*args):
        line = " ".join(str(a) for a in args)
        lines.append(line)
        print(line)

    # ── (A) Vertex profiles text ──────────────────────────────────────────────
    pr(f"\n{'='*72}")
    pr(f"  ARCHETYPE VERTEX PROFILES  (Pearson r of mixture weight vs stat)")
    pr(f"{'='*72}")
    header = f"{'Statistic':<22}" + "".join(f"  {lbl:>14}" for lbl in vertex_labels)
    pr(header)
    pr("-" * len(header))
    for si, sname in enumerate(stat_names):
        row = f"{sname:<22}"
        for k in range(K):
            r, p = r_vertex[si, k], p_vertex[si, k]
            row += f"  {'n/a':>14}" if np.isnan(r) else f"  {r:+.3f}{STARS(p):3s}  "
        pr(row)

    # Build per-vertex hover: top 2 positive and top 2 negative significant
    vertex_hover = [""] * K
    for k in range(K):
        rs = [(r_vertex[si, k], stat_names[si]) for si in range(n_stats)
              if not np.isnan(r_vertex[si, k])]
        rs.sort(reverse=True)
        pos = [f"+{r:.2f} {s}" for r, s in rs if r > 0][:2]
        neg = [f"{r:.2f} {s}" for r, s in rs if r < 0][-2:][::-1]
        vertex_hover[k] = "<br>".join(pos + neg)

    # ── (B) Edge-walk text ────────────────────────────────────────────────────
    pr(f"\n{'='*72}")
    pr(f"  EDGE-WALK GRADIENTS  (Pearson r of position-along-edge vs stat)")
    pr(f"  Position 0 = first archetype,  1 = second archetype")
    pr(f"{'='*72}")
    header2 = f"{'Statistic':<22}" + "".join(f"  {lbl:>10}" for lbl in edge_labels)
    pr(header2)
    pr("-" * len(header2))
    for si, sname in enumerate(stat_names):
        row = f"{sname:<22}"
        for ei in range(n_edges):
            r, p = r_edge[si, ei], p_edge[si, ei]
            row += f"  {'n/a':>10}" if np.isnan(r) else f"  {r:+.3f}{STARS(p):3}"
        pr(row)
    pr("\n  * p<0.05  ** p<0.01  *** p<0.001")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  -> {out_txt}")

    # ── (C) Visual figure ─────────────────────────────────────────────────────
    _archetype_report_figure(
        stat_names, vertex_labels, edge_labels,
        r_vertex, p_vertex, r_edge, p_edge,
        out_txt.with_suffix(".png"),
    )

    return vertex_hover


def _tetrahedron_traces(A: np.ndarray, vertex_hover: list[str] | None = None):
    """
    Build Plotly traces for a tetrahedron defined by 4 vertices A (4×3):
      - a faint semi-transparent Mesh3d for the 4 triangular faces
      - three layered Scatter3d traces simulating a glowing outline on vertices

    vertex_hover: optional list of 4 HTML strings shown in the vertex hover tooltip.
    """
    import plotly.graph_objects as go

    x, y, z = A[:, 0], A[:, 1], A[:, 2]
    labels   = ["A1", "A2", "A3", "A4"]
    hover    = vertex_hover or [""] * 4
    custom   = [f"<b>%{{text}}</b><br>{h}<extra></extra>" for h in hover]

    # Tetrahedral face connectivity (all 4 triangles)
    i_idx = [0, 0, 0, 1]
    j_idx = [1, 1, 2, 2]
    k_idx = [2, 3, 3, 3]

    traces = [
        # ── faces: very faint white fill ─────────────────────────────────
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i_idx, j=j_idx, k=k_idx,
            color="white", opacity=0.06,
            flatshading=True, showlegend=False, hoverinfo="skip",
            name="Archetype simplex",
        ),

        # ── vertex glow layer 1: large outer halo ────────────────────────
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=32, color="#00e5ff", opacity=0.04),
            showlegend=False, hoverinfo="skip",
        ),
        # ── vertex glow layer 2: mid glow ────────────────────────────────
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=18, color="#00e5ff", opacity=0.12),
            showlegend=False, hoverinfo="skip",
        ),
        # ── vertex core + label ───────────────────────────────────────────
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers+text",
            marker=dict(
                size=9, color="white", opacity=1.0,
                line=dict(width=4, color="#00e5ff"),  # ← glowing cyan outline
            ),
            text=labels,
            textfont=dict(color="#00e5ff", size=13),
            textposition="top center",
            name="Archetypes",
            hovertemplate=custom,
        ),
    ]
    return traces


def plot_pca_3d(stems, embeddings, out, site_map=None, notes_map=None,
                renders_dir: Path | None = None, expand: float = 0.20):
    try:
        import plotly  # noqa: F401
    except ImportError:
        log.warning("plotly not installed; skipping 3D PCA.  pip install plotly")
        return
    coords, var, _ = _scale_and_reduce(embeddings, n_components=10)
    coords3 = coords[:, :3]
    axis_labels = [f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})", f"PC3 ({var[2]:.1%})"]

    fig = _plotly_3d(stems, coords3, axis_labels,
                     f"6-view DINOv2 — PCA 3D  ({len(stems)} stones)",
                     site_map, notes_map)

    # ── Fit Pareto archetypes (K=4 tetrahedron) ──────────────────────────────
    log.info("  fitting archetypes (K=4)…")
    Z, S = _fit_archetypes(coords3, K=4)

    # Contract slightly toward centroid so the visual tetrahedron is not at the
    # very edge of the point cloud (Frank-Wolfe anchors vertices to data extremes).
    Z = _expand_archetypes(Z, factor=-0.15)
    n_in = _inside_tetrahedron(coords3, Z)
    log.info(f"  {n_in}/{len(stems)} stones inside tetrahedron")

    # Save for dashboard use (3D PCA panel + vertex click detail)
    np.save(out / "archetypes_Z.npy", Z)   # (4, 3) vertex positions in PCA-3D space
    np.save(out / "archetypes_S.npy", S)   # (N, 4) mixture weights per stone

    # ── Correlation report (vertex profiles + edge-walk gradients) ───────────
    vertex_hover = None
    if renders_dir is not None and renders_dir.exists():
        stat_names, stat_mat = _pz_stats(stems, renders_dir)
        vertex_hover = _archetype_report(
            stems, coords3, S, Z, stat_names, stat_mat,
            out / "archetype_report.txt",
        )

    for trace in _tetrahedron_traces(Z, vertex_hover):
        fig.add_trace(trace)

    html = out / "pca_3d.html"
    fig.write_html(str(html))
    log.info(f"  -> {html}")


def plot_umap(stems, embeddings, out, site_map=None, notes_map=None, seed=42):
    try:
        import umap as _umap
    except ImportError:
        log.warning("umap-learn not installed; skipping UMAP.  pip install umap-learn")
        return
    from sklearn.preprocessing import StandardScaler
    E    = StandardScaler().fit_transform(np.stack(embeddings))
    n_nb = min(15, len(stems) - 1)
    c2   = _umap.UMAP(n_components=2, n_neighbors=n_nb, random_state=seed).fit_transform(E)
    _plot_2d(stems, c2, site_map, notes_map,
             f"6-view DINOv2 — UMAP  ({len(stems)} stones)",
             "UMAP-1", "UMAP-2", out / "umap.png")
    try:
        import plotly  # noqa: F401
        c3 = _umap.UMAP(n_components=3, n_neighbors=n_nb, random_state=seed).fit_transform(E)
        fig3 = _plotly_3d(stems, c3, ["UMAP-1", "UMAP-2", "UMAP-3"],
                          f"6-view DINOv2 — UMAP 3D  ({len(stems)} stones)",
                          site_map, notes_map)
        html = out / "umap_3d.html"
        fig3.write_html(str(html))
        log.info(f"  -> {html}")
    except ImportError:
        pass


def _run_plots(stems, embeddings, out, site_map, notes_map, seed,
               renders_dir: Path | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plot_pca(stems, embeddings, out, site_map, notes_map)
            plot_pca_13(stems, embeddings, out, site_map, notes_map)
            plot_pca_3d(stems, embeddings, out, site_map, notes_map,
                        renders_dir=renders_dir)
            plot_umap(stems, embeddings, out, site_map, notes_map, seed=seed)
            plot_similarity_matrix(stems, embeddings, out, site_map)
        except Exception:
            log.debug(f"Plot skipped (n={len(stems)}): {traceback.format_exc()[-200:]}")


# ---------------------------------------------------------------------------
# Mesh processing
# ---------------------------------------------------------------------------

def _process_mesh(path: Path, args) -> tuple[np.ndarray, np.ndarray] | None:
    raw = load_mesh(path)
    v, f = raw["vertices"], raw["faces"]
    log.info(f"  loaded  {len(v):>8,}V  {len(f):>8,}F")
    if not args.skip_clean:
        v, f, meta = clean_mesh(v, f, keep_largest_component=True)
        log.info(f"  clean   {meta['num_clean_vertices']:>8,}V  {meta['num_clean_faces']:>8,}F")
    if len(f) == 0:
        log.warning("  No faces — skipped")
        return None
    if args.decimate:
        from src.decimation import decimate
        from src.mesh_cleaning import orient_normals_outward
        tf = args.target_faces or max(500, int(len(f) * args.target_ratio))
        try:
            v, f = decimate(v, f, tf, method=args.decimate_method)
            f    = orient_normals_outward(v, f)
            log.info(f"  decimate {len(v):>8,}V  {len(f):>8,}F  (target {tf})")
        except Exception as e:
            log.warning(f"  Decimate failed ({e}), continuing with full mesh")
    return v, f


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wrl_dir",         default=str(WRL_DIR))
    p.add_argument("--output_dir",      default=str(OUTPUT_BASE),
                   help="Base output folder; a timestamp subfolder is created inside")
    p.add_argument("--site_xlsx",       default=str(XLSX_DEFAULT))
    p.add_argument("--no_site_color",   action="store_true")
    p.add_argument("--dino_model",      default=DINO_MODEL,
                   help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14")
    p.add_argument("--image_size",      type=int, default=IMAGE_SIZE)
    p.add_argument("--device",          default=None)
    p.add_argument("--reload_scale",    default=None, metavar="JSON",
                   help="Path to a saved multiview_scale.json; skip pass-1 scan")
    p.add_argument("--limit",           type=int, default=None,
                   help="Process only the first N WRL files")
    p.add_argument("--skip_clean",      action="store_true")
    p.add_argument("--decimate",        action="store_true")
    p.add_argument("--target_faces",    type=int, default=None)
    p.add_argument("--target_ratio",    type=float, default=0.05)
    p.add_argument("--decimate_method", default="qem", choices=["qem", "cluster"])
    p.add_argument("--renders_dir",     default=str(RENDERS_DIR))
    p.add_argument("--pca_only",        action="store_true",
                   help="Skip rendering; reload embeddings from an existing run and replot")
    p.add_argument("--run_dir",         default=None, metavar="DIR",
                   help="Stamped run folder for --pca_only (default: most recent)")
    p.add_argument("--seed",            type=int, default=42)
    # SimCLR finetuning (optional)
    p.add_argument("--finetune",        type=int, nargs="?", const=10, default=None,
                   metavar="EPOCHS",
                   help="Run SimCLR finetuning; optionally set epoch count (default 10)")
    p.add_argument("--ft_blocks",       type=int, default=0,
                   help="Backbone blocks to unfreeze (0=head only, -1=all)")
    p.add_argument("--ft_lr",           type=float, default=1e-4)
    p.add_argument("--ft_batch",        type=int, default=16)
    p.add_argument("--ft_temp",         type=float, default=0.07)
    return p.parse_args()


def _load_site_and_notes(args):
    if args.no_site_color:
        return None, None
    xlsx = Path(args.site_xlsx)
    if not xlsx.exists():
        log.warning(f"site_xlsx not found: {xlsx}")
        return None, None
    try:
        return load_site_map(xlsx)
    except Exception as e:
        log.warning(f"Could not load site map: {e}")
        return None, None


def _latest_run(base: Path) -> Path | None:
    runs = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    return runs[0] if runs else None


def main():
    args     = parse_args()
    site_map, notes_map = _load_site_and_notes(args)
    base_dir = Path(args.output_dir)

    if args.pca_only:
        out = Path(args.run_dir) if args.run_dir else _latest_run(base_dir)
        if out is None:
            log.error(f"No run folders found under {base_dir}")
            return
        files = sorted((out / "embeddings").glob("*.npy"))
        if not files:
            log.error(f"No embeddings in {out / 'embeddings'}")
            return
        stems      = [f.stem for f in files]
        embeddings = [np.load(f) for f in files]
        log.info(f"Replotting {len(stems)} embeddings from {out}")
        _run_plots(stems, embeddings, out, site_map, notes_map, args.seed,
                   renders_dir=Path(args.renders_dir))
        return

    device  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out     = ensure_dir(base_dir / stamp)
    emb_dir = ensure_dir(out / "embeddings")

    files = collect_mesh_files(args.wrl_dir, limit=args.limit)
    if not files:
        log.error(f"No WRL files found in {args.wrl_dir}")
        return
    log.info(f"Found {len(files)} WRL files  →  {out}")

    # Pass 1 — global extents
    if args.reload_scale:
        max_x, max_y, max_z = load_scale(Path(args.reload_scale))
    else:
        max_x, max_y, max_z = find_global_params(files)
        save_scale(max_x, max_y, max_z)

    model = build_dino(args.dino_model, device=device)
    log.info(f"embed_dim={model.embed_dim}  device={device}")

    renders_dir = ensure_dir(Path(args.renders_dir))
    stems: list[str] = []

    # Pass 2 — render 6 depth views per stone (cached)
    for i, path in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {path.stem}")
        if (renders_dir / f"{path.stem}_pZ.png").exists():
            log.info("  render cached — skipping")
            stems.append(path.stem)
            continue
        try:
            result = _process_mesh(path, args)
            if result is None:
                continue
            v, f = result
            views = render_multiview(v, f, max_x, max_y, max_z, args.image_size)
            save_multiview(views, path.stem, renders_dir)
            stems.append(path.stem)
        except Exception:
            log.error(f"  Failed:\n{traceback.format_exc()[-400:]}")

    if not stems:
        log.error("No renders produced.")
        return

    # Pass 3 (optional) — SimCLR finetuning
    if args.finetune is not None:
        all_views = [load_multiview(s, renders_dir) for s in stems]
        model     = finetune_simclr(
            model, all_views, device,
            n_epochs=args.finetune, lr=args.ft_lr,
            batch_size=args.ft_batch, n_unfreeze_blocks=args.ft_blocks,
            temperature=args.ft_temp, img_size=args.image_size,
        )
        ckpt_path = out / "finetune_checkpoint.pt"
        torch.save(model.state_dict(), ckpt_path)
        log.info(f"Checkpoint saved → {ckpt_path}")

    # Pass 4 — embed (mean-pool 6 views)
    embeddings: list[np.ndarray] = []
    for stem in stems:
        views = load_multiview(stem, renders_dir)
        e     = embed_multiview(model, views, device, args.image_size)
        np.save(emb_dir / f"{stem}.npy", e)
        embeddings.append(e)
        log.info(f"  {stem}  dim={e.shape[0]}")

    all_E = np.stack(embeddings)
    np.save(out / "all_embeddings.npy", all_E)
    log.info(f"Done. {len(stems)} embeddings  shape={all_E.shape}  output: {out}")

    if len(stems) >= 2:
        _run_plots(stems, embeddings, out, site_map, notes_map, args.seed,
                   renders_dir=renders_dir)


if __name__ == "__main__":
    main()
