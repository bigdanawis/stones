"""Shared utilities: logging, metadata I/O, file helpers."""
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "filename",
    "num_original_vertices",
    "num_original_faces",
    "num_clean_vertices",
    "num_clean_faces",
    "sampling_mode",
    "num_points",
    "centroid_x", "centroid_y", "centroid_z",
    "scale_factor",
    "bbox_x", "bbox_y", "bbox_z",
    "surface_area",
    "embedding_path",
    "pointcloud_path",
    "preview_path",
]

FAILURE_FIELDS = ["filename", "stage", "error"]


class MetadataWriter:
    """Append-safe writer for per-mesh metadata rows."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._init_csv(self.path, METADATA_FIELDS)

    @staticmethod
    def _init_csv(path: Path, fields: List[str]):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(self, row: Dict[str, Any]):
        row_padded = {k: row.get(k, "") for k in METADATA_FIELDS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=METADATA_FIELDS).writerow(row_padded)


class FailureWriter:
    """Append-safe writer for failed-mesh rows."""

    def __init__(self, path: Path):
        self.path = Path(path)
        MetadataWriter._init_csv(self.path, FAILURE_FIELDS)

    def write(self, filename: str, stage: str, error: str):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FAILURE_FIELDS).writerow(
                {"filename": filename, "stage": stage, "error": error}
            )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._t = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t

    def reset(self):
        self._t = time.perf_counter()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def collect_mesh_files(directory: str, limit: Optional[int] = None) -> List[Path]:
    exts = {".wrl", ".vrml"}
    paths = sorted(
        p for p in Path(directory).rglob("*") if p.suffix.lower() in exts
    )
    if limit:
        paths = paths[:limit]
    return paths


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
