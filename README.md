# Stone-Tool Point-MAE Pipeline

Processes directories of WRL/VRML 3D scan meshes (archaeological lithic tools)
and extracts Point-MAE embeddings for downstream analysis.

## Directory layout

```
stones/
├── requirements.txt
├── src/
│   ├── mesh_io.py          # WRL loading (trimesh → meshio → native parser)
│   ├── mesh_cleaning.py    # duplicate removal, normals, largest component
│   ├── sampling.py         # uniform / curvature / edge_aware sampling
│   ├── pointmae_embedder.py# Point-MAE encoder + checkpoint loading
│   └── utils.py            # logging, metadata CSV, file helpers
└── scripts/
    ├── extract_pointmae_embeddings.py  # main CLI pipeline
    ├── check_mesh_loading.py           # diagnostic: test loading + sampling
    └── visualize_embeddings.py         # PCA / UMAP / nearest-neighbour viz
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Check that your WRL files load correctly
python scripts/check_mesh_loading.py --dir data/wrl --limit 3

# 2. Run the full pipeline
python scripts/extract_pointmae_embeddings.py \
    --input_dir  data/wrl \
    --output_dir outputs/run1 \
    --checkpoint path/to/pointmae_pretrain.pth \
    --num_points 4096 \
    --sampling_mode edge_aware \
    --save_previews

# 3. Visualize embeddings
python scripts/visualize_embeddings.py \
    --embeddings outputs/run1/all_embeddings.npy \
    --metadata   outputs/run1/metadata.csv \
    --output_dir outputs/run1/viz \
    --umap
```

## CLI reference — extract_pointmae_embeddings.py

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | required | Directory with `.wrl` / `.vrml` files |
| `--output_dir` | required | Root for all output files |
| `--checkpoint` | None | Path to pretrained Point-MAE `.pth` checkpoint |
| `--num_points` | 4096 | Surface points to sample per mesh |
| `--sampling_mode` | `edge_aware` | `uniform` / `curvature` / `edge_aware` |
| `--edge_ratio` | 0.30 | Fraction from high-dihedral zone (`edge_aware` only) |
| `--device` | auto | `cuda` or `cpu` |
| `--limit` | None | Process at most N files (testing) |
| `--save_previews` | off | Write PLY point-cloud preview per mesh |
| `--embed_dim` | 384 | Embedding dimension (384 = ViT-Small, 768 = ViT-Base) |
| `--num_groups` | 64 | Number of patch groups G |
| `--group_size` | 32 | Points per group K |
| `--seed` | 42 | Random seed |

## Outputs

```
outputs/run1/
├── metadata.csv             # per-mesh geometry + path metadata
├── failed_meshes.csv        # files that failed with error stage + message
├── all_embeddings.npy       # [N, D] stacked embeddings
├── all_embeddings.parquet   # same, with filename index
├── embeddings/
│   └── <stem>.npy           # [D] embedding per mesh
├── pointclouds/
│   └── <stem>.npy           # [num_points, 6] XYZ+normal per mesh
└── previews/
    └── <stem>.ply           # ASCII PLY point cloud (with --save_previews)
```

## Point-MAE integration

### Using a pretrained checkpoint (standalone)

The standalone encoder (`StandalonePointMAE`) re-implements the Point-MAE
ViT-Small architecture in pure PyTorch and loads weights directly:

```bash
python scripts/extract_pointmae_embeddings.py \
    --checkpoint pointmae_pretrain.pth \
    ...
```

The loader tries these state-dict keys in order:
`base_model`, `model`, `encoder`, `state_dict`, `model_state_dict`.
It also strips `module.`, `MAE_encoder.` and `encoder.` prefixes automatically.

### Using the official Point-MAE repo

Set the `POINTMAE_REPO` environment variable to the repo root:

```bash
export POINTMAE_REPO=/path/to/Point-MAE
python scripts/extract_pointmae_embeddings.py --checkpoint ...
```

The wrapper imports `models.Point_MAE` from that path. **If the class name or
config keys differ in your version**, edit the `# <<ADJUST>>` lines in
[src/pointmae_embedder.py](src/pointmae_embedder.py).

### Swapping in a different encoder (Point-M2AE, etc.)

Subclass `BaseEmbedder` and implement `embed(points) -> np.ndarray`:

```python
from src.pointmae_embedder import BaseEmbedder

class MyEncoder(BaseEmbedder):
    def __init__(self, checkpoint, device):
        ...
    def embed(self, points):       # [N, 3] → [D]
        ...
```

## Sampling modes

| Mode | Behaviour | Best for |
|------|-----------|----------|
| `uniform` | Area-weighted random sampling | Baseline / ablation |
| `curvature` | Bias toward high-curvature regions (ridges, tips) | Capturing fine detail |
| `edge_aware` | 70% area-weighted + 30% high-dihedral edges | **Recommended for lithics** |

The 70/30 split is controlled by `--edge_ratio`.

## Visualization

```bash
# PCA + nearest neighbours (always)
python scripts/visualize_embeddings.py \
    --embeddings outputs/run1/all_embeddings.npy \
    --metadata   outputs/run1/metadata.csv \
    --n_neighbors 5

# Add UMAP (needs: pip install umap-learn)
    --umap

# Colour scatter by a metadata column (e.g. surface_area)
    --color_by surface_area
```

Outputs: `pca.png`, `pca_coords.csv`, `umap.png`, `umap_coords.csv`,
`nearest_neighbors.csv`.
