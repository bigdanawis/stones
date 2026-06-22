"""
Point-MAE encoder wrapper.

Loading strategy (tried in order):
  1. Official Point-MAE repo — if POINTMAE_REPO env var is set, or if the
     repo is importable as `models.Point_MAE`, import the model class and
     load the checkpoint directly.
  2. Standalone re-implementation — a self-contained ViT-Small encoder that
     matches Point-MAE's default architecture (G=64 groups, K=32 neighbours,
     embedding dim 384) so existing checkpoints transfer weight-for-weight.

To swap in Point-M2AE or another encoder:
  - Subclass `BaseEmbedder` and override `_build_model` + `_extract_embedding`.
  - Pass the subclass to the pipeline via --encoder_class.

Checkpoint format assumed:
  torch.save({'base_model': model.state_dict(), ...}, path)
  or
  torch.save(model.state_dict(), path)
"""
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_embedder(
    checkpoint: Optional[str],
    device: str,
    num_points: int = 4096,
    group_size: int = 32,
    num_groups: int = 64,
    embed_dim: int = 384,
) -> "BaseEmbedder":
    """
    Return a ready-to-use embedder.
    Tries the official Point-MAE repo first; falls back to standalone impl.
    """
    embedder = None

    if checkpoint is not None:
        embedder = _try_official_pointmae(checkpoint, device, num_points)

    if embedder is None:
        log.info("Using standalone Point-MAE encoder.")
        embedder = StandalonePointMAE(
            checkpoint=checkpoint,
            device=device,
            num_points=num_points,
            group_size=group_size,
            num_groups=num_groups,
            embed_dim=embed_dim,
        )

    return embedder


def _try_official_pointmae(checkpoint: str, device: str, num_points: int):
    """
    Attempt to import the official Point-MAE repo.
    Returns an OfficialPointMAE wrapper or None if import fails.
    """
    repo_path = os.environ.get("POINTMAE_REPO", "")
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        # Official repo exposes the model under models.Point_MAE or similar.
        # ADJUST the import below to match your repo version.
        import models.Point_MAE as pm_module  # noqa: PLC0415
        log.info(f"Found official Point-MAE repo at {repo_path or 'sys.path'}")
        return OfficialPointMAE(pm_module, checkpoint, device)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder:
    def embed(self, points: np.ndarray) -> np.ndarray:
        """
        points : [N, 3] or [N, 6] float32 numpy array
        Returns : [D] float32 embedding vector
        """
        raise NotImplementedError

    def embed_batch(self, points_list: list) -> np.ndarray:
        return np.stack([self.embed(p) for p in points_list])


# ---------------------------------------------------------------------------
# Wrapper for the official Point-MAE repo
# ---------------------------------------------------------------------------

class OfficialPointMAE(BaseEmbedder):
    """
    Thin wrapper around the official Point-MAE model.

    ADJUST the lines marked # <<ADJUST>> to match your checkpoint layout.
    """
    def __init__(self, pm_module, checkpoint: str, device: str):
        self.device = torch.device(device)

        # <<ADJUST>> Build the encoder config matching the pretrained checkpoint.
        # Typical Point-MAE ViT-Small config:
        from easydict import EasyDict  # often used in the official repo
        cfg = EasyDict({
            "trans_dim": 384,
            "depth": 12,
            "drop_path_rate": 0.1,
            "cls_dim": 40,           # doesn't matter for embedding extraction
            "num_heads": 6,
            "group_size": 32,
            "num_group": 64,
            "encoder_dims": 256,
        })

        # <<ADJUST>> Instantiate the model class from the official repo.
        self.model = pm_module.PointMAE(cfg)
        self.model = self.model.to(self.device)
        self._load_checkpoint(checkpoint)
        self.model.eval()
        log.info("Official Point-MAE encoder loaded.")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # <<ADJUST>> Key name may differ (base_model, model, state_dict, …)
        state = ckpt.get("base_model", ckpt.get("model", ckpt))
        if isinstance(state, dict):
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                log.warning(f"Missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
            if unexpected:
                log.warning(f"Unexpected keys: {unexpected[:5]}")
        else:
            log.warning("Could not extract state_dict from checkpoint.")

    @torch.no_grad()
    def embed(self, points: np.ndarray) -> np.ndarray:
        pts = torch.from_numpy(points[:, :3]).float().unsqueeze(0).to(self.device)

        # <<ADJUST>> Access the encoder output before the classification head.
        # In the official repo, self.model.MAE_encoder returns token sequences.
        # We mean-pool over token dimension to get the global embedding.
        tokens = self.model.MAE_encoder(pts)  # [1, G, D]
        embedding = tokens.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding


# ---------------------------------------------------------------------------
# Standalone Point-MAE encoder (no repo dependency)
# ---------------------------------------------------------------------------

class StandalonePointMAE(BaseEmbedder):
    """
    Self-contained re-implementation of the Point-MAE ViT-Small encoder.

    Architecture:
      1. Farthest Point Sampling → G centre points
      2. KNN grouping            → G groups of K points
      3. Mini-PointNet per group → G tokens of dim `embed_dim`
      4. Transformer encoder     → contextualised tokens
      5. Mean pool               → global embedding [embed_dim]

    Default hyper-parameters match the ViT-Small pretrained checkpoint.
    """

    def __init__(
        self,
        checkpoint: Optional[str],
        device: str,
        num_points: int = 4096,
        group_size: int = 32,    # K
        num_groups: int = 64,    # G
        embed_dim: int = 384,    # transformer width
        depth: int = 12,
        num_heads: int = 6,
        drop_path: float = 0.1,
    ):
        self.device = torch.device(device)
        self.num_groups = num_groups
        self.group_size = group_size

        self.model = _PointMAEEncoder(
            group_size=group_size,
            num_groups=num_groups,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path,
        ).to(self.device)

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
        else:
            log.warning(
                "No checkpoint provided — embeddings will be random (encoder not pretrained)."
            )

        self.model.eval()

    def _load_checkpoint(self, path: str):
        log.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        for key in ("base_model", "model", "encoder", "state_dict", "model_state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt

        remapped = {}
        for k, v in state.items():
            # Strip DataParallel / repo-specific prefixes
            k = k.replace("module.", "")

            # Official Point-MAE repo: transformer blocks live at
            # blocks.blocks.N.* — remap to our blocks.N.*
            import re as _re
            k = _re.sub(r"^blocks\.blocks\.", "blocks.", k)

            # Attention weight name differences:
            #   official: attn.qkv.*  →  ours: attn.in_proj_*
            #   official: attn.proj.* →  ours: attn.out_proj.*
            k = k.replace(".attn.qkv.weight", ".attn.in_proj_weight")
            k = k.replace(".attn.qkv.bias",   ".attn.in_proj_bias")
            k = k.replace(".attn.proj.",       ".attn.out_proj.")

            # MLP name differences:
            #   official: mlp.fc1.* / mlp.fc2.*  →  ours: mlp.0.* / mlp.2.*
            k = k.replace(".mlp.fc1.", ".mlp.0.")
            k = k.replace(".mlp.fc2.", ".mlp.2.")

            # Strip remaining repo-specific prefixes
            k = k.replace("MAE_encoder.", "").replace("encoder.", "")

            remapped[k] = v

        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        loaded = len(remapped) - len(unexpected)
        log.info(f"Checkpoint loaded: {loaded}/{len(remapped)} weights matched "
                 f"({len(missing)} missing, {len(unexpected)} unexpected)")

    @torch.no_grad()
    def embed(self, points: np.ndarray) -> np.ndarray:
        pts = torch.from_numpy(points[:, :3]).float().unsqueeze(0).to(self.device)
        embedding = self.model(pts)          # [1, embed_dim]
        return embedding.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# _PointMAEEncoder — the actual PyTorch module
# ---------------------------------------------------------------------------

class _PointMAEEncoder(nn.Module):
    def __init__(
        self,
        group_size: int = 32,
        num_groups: int = 64,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        drop_path_rate: float = 0.1,
        encoder_hidden: int = 256,
    ):
        super().__init__()
        self.group_size = group_size
        self.num_groups = num_groups

        # 1. Mini-PointNet tokeniser
        self.tokeniser = _MiniPointNet(3, encoder_hidden, embed_dim)

        # 2. Learned positional encoding (per-centre-point)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim)
        )

        # 3. Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: [B, N, 3] → [B, embed_dim]"""
        return self.forward_tokens(pts)[0].mean(dim=1)

    def forward_tokens(self, pts: torch.Tensor, ids_visible=None):
        """
        Full grouping + transformer pass.

        pts         : [B, N, 3]
        ids_visible : optional [B, G'] index tensor to process only a subset of groups
                      (used for MAE fine-tuning where masked groups are withheld).

        Returns (tokens [B, G', D], centres [B, G, 3])
        """
        B, N, _ = pts.shape

        centres = _fps(pts, self.num_groups)                        # [B, G, 3]
        groups  = _knn_group(pts, centres, self.group_size)         # [B, G, K, 3]

        # Normalise each group to its centre
        groups_local = groups - centres.unsqueeze(2)                # [B, G, K, 3]

        B, G, K, _ = groups_local.shape
        tokens = self.tokeniser(groups_local.reshape(B * G, K, 3)) # [B*G, D]
        tokens = tokens.reshape(B, G, -1)                          # [B, G, D]

        pos = self.pos_embed(centres)                               # [B, G, D]
        x = tokens + pos                                            # [B, G, D]

        if ids_visible is not None:
            D = x.shape[-1]
            x = x.gather(1, ids_visible.unsqueeze(-1).expand(-1, -1, D))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x, centres                                           # [B, G', D], [B, G, 3]


class _MiniPointNet(nn.Module):
    """Shared MLP + global max-pool over K neighbours."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [M, K, 3] → [M, out_dim]"""
        M, K, C = x.shape
        x = x.reshape(M * K, C)
        x = self.net(x)
        x = x.reshape(M, K, -1)
        x = x.max(dim=1).values   # global max pool over K
        return self.proj(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )
        # Stochastic depth
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


# ---------------------------------------------------------------------------
# Geometry ops: FPS and KNN (pure PyTorch, no open3d / torch-cluster needed)
# ---------------------------------------------------------------------------

def _fps(pts: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest Point Sampling. pts: [B, N, 3] → [B, S, 3]"""
    B, N, _ = pts.shape
    device = pts.device
    idx = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(num_samples):
        idx[:, i] = farthest
        centroid = pts[torch.arange(B), farthest].unsqueeze(1)  # [B, 1, 3]
        d = ((pts - centroid) ** 2).sum(dim=-1)                  # [B, N]
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=1)

    return pts[torch.arange(B).unsqueeze(1), idx]               # [B, S, 3]


def _knn_group(pts: torch.Tensor, centres: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each centre, find its K nearest neighbours in `pts`.
    pts     : [B, N, 3]
    centres : [B, G, 3]
    returns : [B, G, K, 3]
    """
    B, N, _ = pts.shape
    G = centres.shape[1]

    # Pairwise squared distances [B, G, N]
    pts_sq     = (pts ** 2).sum(dim=-1, keepdim=True)           # [B, N, 1]
    centres_sq = (centres ** 2).sum(dim=-1, keepdim=True)       # [B, G, 1]
    cross      = torch.bmm(centres, pts.transpose(1, 2))         # [B, G, N]
    dists_sq   = centres_sq + pts_sq.transpose(1, 2) - 2 * cross

    knn_idx = dists_sq.topk(k, dim=-1, largest=False).indices   # [B, G, K]
    # Gather neighbours
    idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)       # [B, G, K, 3]
    pts_exp = pts.unsqueeze(1).expand(-1, G, -1, -1)            # [B, G, N, 3]
    return pts_exp.gather(2, idx_exp)                            # [B, G, K, 3]
