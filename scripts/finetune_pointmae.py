"""
Fine-tune Point-MAE on stone tool point clouds.

Modes
-----
  mae       (default) Self-supervised masked-autoencoder continued pre-training.
            No labels needed.  Masks 60% of point groups; trains encoder +
            lightweight MLP decoder to predict masked patch centres.

  classify  Supervised using site labels from the Excel spreadsheet.
            Freezes the first --freeze_blocks transformer blocks and trains
            the rest + a classification head.

Usage
-----
  # self-supervised (recommended first step)
  python scripts/finetune_pointmae.py

  # supervised with site labels
  python scripts/finetune_pointmae.py --mode classify --epochs 100

  # resume from a previous fine-tune
  python scripts/finetune_pointmae.py --checkpoint pth/finetuned_stones_20260620.pth
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pointmae_embedder import _PointMAEEncoder, _TransformerBlock, _fps, _knn_group
from src.utils import get_logger

log = get_logger("finetune")

PTH_DIR     = ROOT / "pth"
WRL_DIR     = ROOT / "wrl"
XLSX_DEFAULT = WRL_DIR / "Handaxes 2026 list with sites.xlsx"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PointCloudDataset(Dataset):
    def __init__(self, files, labels=None, num_points=2048, augment=True):
        self.files     = files
        self.labels    = labels      # None (MAE) or list of int (classify)
        self.num_points = num_points
        self.augment   = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])          # [N, 3] or [N, 6]
        pts  = data[:, :3].astype(np.float32)

        N   = len(pts)
        sel = np.random.choice(N, self.num_points,
                               replace=(N < self.num_points))
        pts = pts[sel]

        if self.augment:
            pts = _augment(pts)

        t = torch.from_numpy(pts)
        if self.labels is not None:
            return t, torch.tensor(self.labels[idx], dtype=torch.long)
        return t


def _augment(pts: np.ndarray) -> np.ndarray:
    # Random rotation around the up axis (Z)
    angle = np.random.uniform(0, 2 * np.pi)
    c, s  = float(np.cos(angle)), float(np.sin(angle))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    # Gaussian jitter
    pts += np.random.normal(0, 0.005, pts.shape).astype(np.float32)
    # Random scale ±10%
    pts *= np.random.uniform(0.9, 1.1)
    return pts


# ---------------------------------------------------------------------------
# MAE fine-tuner module
# ---------------------------------------------------------------------------

def _chamfer(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Chamfer distance between two point sets.
    pred, target : [M, K, 3]
    """
    # [M, K_pred, K_tgt]
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(-1)                         # [M, Kp, Kt]
    loss = dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()
    return loss


class MAEFineTuner(nn.Module):
    """
    Wraps _PointMAEEncoder for MAE self-supervised fine-tuning.

    Decoder design follows the original Point-MAE paper:
      1. Encode only the visible patches.
      2. Insert mask tokens at masked positions, each with its patch's
         positional encoding so the decoder knows WHERE to predict.
      3. Run a small transformer decoder over the full visible+masked sequence
         so each mask token can attend to the encoded visible context.
      4. Predict K local points per masked patch; optimise with Chamfer distance.

    Without steps 2+3 the decoder has no spatial context and immediately
    collapses to predicting the average local patch shape (loss ~0.0007, flat).
    """
    def __init__(self, encoder: _PointMAEEncoder, mask_ratio=0.60,
                 decoder_dim=256, decoder_depth=4):
        super().__init__()
        self.encoder    = encoder
        self.mask_ratio = mask_ratio
        K = encoder.group_size
        D = encoder.norm.normalized_shape[0]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Small transformer decoder — attends visible tokens, queries mask positions
        self.decoder_blocks = nn.ModuleList([
            _TransformerBlock(D, heads=6) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(D)

        # Final MLP: D → K*3 local coords
        self.decoder_pred = nn.Sequential(
            nn.Linear(D, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, K * 3),
        )
        self.K = K

    def forward(self, pts: torch.Tensor):
        """pts: [B, N, 3]  →  loss scalar"""
        B, N, _  = pts.shape
        enc      = self.encoder
        G        = enc.num_groups
        K        = enc.group_size
        n_mask   = int(G * self.mask_ratio)
        n_vis    = G - n_mask

        # Grouping — need patch points for the reconstruction target
        centres      = _fps(pts, G)                         # [B, G, 3]
        groups       = _knn_group(pts, centres, K)          # [B, G, K, 3]
        groups_local = groups - centres.unsqueeze(2)        # local coords

        # Random masking
        noise    = torch.rand(B, G, device=pts.device)
        order    = noise.argsort(dim=1)
        ids_vis  = order[:, n_mask:]                        # [B, n_vis]
        ids_mask = order[:, :n_mask]                        # [B, n_mask]

        # Encode only visible patches
        vis_tokens, _ = enc.forward_tokens(pts, ids_visible=ids_vis)  # [B, n_vis, D]
        D = vis_tokens.shape[-1]

        # Positional encodings for all patches
        pos_all = enc.pos_embed(centres)                    # [B, G, D]

        # Build full decoder input sequence:
        #   visible positions  → encoded token + its positional encoding
        #   masked positions   → learned mask token + its positional encoding
        pos_vis  = pos_all.gather(1, ids_vis.unsqueeze(-1).expand(-1, -1, D))
        pos_mask = pos_all.gather(1, ids_mask.unsqueeze(-1).expand(-1, -1, D))

        vis_seq  = vis_tokens + pos_vis                                  # [B, n_vis, D]
        mask_seq = self.mask_token.expand(B, n_mask, D) + pos_mask      # [B, n_mask, D]

        # Concatenate [visible | masked] — order doesn't matter for attention
        full_seq = torch.cat([vis_seq, mask_seq], dim=1)                 # [B, G, D]

        for blk in self.decoder_blocks:
            full_seq = blk(full_seq)
        full_seq = self.decoder_norm(full_seq)

        # Extract predictions for masked positions (appended after visible)
        masked_out = full_seq[:, n_vis:]                                 # [B, n_mask, D]
        BM = B * n_mask
        pred = self.decoder_pred(masked_out).reshape(BM, K, 3)

        # Target: actual local patch coords for masked groups
        target = groups_local.gather(
            1, ids_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 3)
        ).reshape(BM, K, 3)

        return _chamfer(pred, target)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, epochs, warmup=5):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for pts in loader:
        pts  = pts.to(device)
        loss = model(pts)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for pts in loader:
        total_loss += model(pts.to(device)).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pc_dir",         default=None,
                   help="Directory of .npy point clouds (default: outputs/demo/pointclouds)")
    p.add_argument("--checkpoint",     default=str(PTH_DIR / "modelnet_8k.pth"))
    p.add_argument("--out_checkpoint", default=None,
                   help="Output path (default: pth/finetuned_stones_YYYYMMDD_HHmmss.pth)")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--mask_ratio",     type=float, default=0.60)
    p.add_argument("--freeze_blocks",  type=int,   default=8,
                   help="Freeze first N transformer blocks")
    p.add_argument("--num_points",     type=int,   default=2048)
    p.add_argument("--val_split",      type=float, default=0.15)
    p.add_argument("--device",         default=None)
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--group_size",     type=int,   default=32)
    p.add_argument("--num_groups",     type=int,   default=64)
    p.add_argument("--embed_dim",      type=int,   default=384)
    p.add_argument("--log_every",      type=int,   default=1)
    return p.parse_args()


def _find_pc_dir() -> Path:
    pc_dir = ROOT / "outputs" / "demo" / "pointclouds"
    if pc_dir.is_dir() and any(pc_dir.glob("*.npy")):
        return pc_dir
    raise FileNotFoundError(
        f"No .npy files found in {pc_dir}. "
        "Run demo_embed_pca.py first, or pass --pc_dir explicitly."
    )


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Device: {device}")

    # Locate point clouds
    pc_dir = Path(args.pc_dir) if args.pc_dir else _find_pc_dir()
    files  = sorted(pc_dir.glob("*.npy"))
    if not files:
        log.error(f"No .npy files found in {pc_dir}")
        return
    log.info(f"Found {len(files)} point clouds in {pc_dir}")

    # Build encoder — reuse StandalonePointMAE which has proper key remapping
    from src.pointmae_embedder import build_embedder as _build
    _wrapper = _build(
        checkpoint=args.checkpoint if Path(args.checkpoint).exists() else None,
        device=str(device),
        num_points=args.num_points,
        group_size=args.group_size,
        num_groups=args.num_groups,
        embed_dim=args.embed_dim,
    )
    encoder = _wrapper.model.to(device)

    # Freeze early blocks to prevent catastrophic forgetting on the small dataset
    for i, blk in enumerate(encoder.blocks):
        if i < args.freeze_blocks:
            for p in blk.parameters():
                p.requires_grad = False
    log.info(f"MAE fine-tuning  mask_ratio={args.mask_ratio}  "
             f"freeze_blocks={args.freeze_blocks}/{len(encoder.blocks)}")

    model   = MAEFineTuner(encoder, mask_ratio=args.mask_ratio).to(device)
    dataset = PointCloudDataset(files, num_points=args.num_points, augment=True)
    log.info(f"  n={len(dataset)}")

    # Train / val split
    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Disable augmentation on val split
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              drop_last=(n_train >= args.batch_size))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, args.epochs, warmup=5)

    best_val_loss = float("inf")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_ckpt = Path(args.out_checkpoint) if args.out_checkpoint \
               else PTH_DIR / f"finetuned_stones_{ts}.pth"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Training for {args.epochs} epochs  "
             f"(train={n_train}, val={n_val}  bs={args.batch_size})")
    log.info(f"Output checkpoint -> {out_ckpt}")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        vl_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"encoder": encoder.state_dict(),
                        "epoch": ep,
                        "val_loss": vl_loss},
                       out_ckpt)

        if ep % args.log_every == 0 or ep == 1 or ep == args.epochs:
            elapsed = time.time() - t0
            log.info(f"[{ep:>4}/{args.epochs}]  "
                     f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}  "
                     f"elapsed={elapsed:.0f}s")

    log.info(f"Done. Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoint saved -> {out_ckpt}")
    log.info(f"To use: pass --checkpoint {out_ckpt} to demo_embed_pca.py")


if __name__ == "__main__":
    main()
