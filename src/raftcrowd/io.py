"""Strict manifests, prospective configuration and provenance without GPU imports."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = {
    "seed": 7, "height": 360, "width": 640, "weights": "C_T_V1", "updates": 12,
    "device": "cuda", "mixed_precision": False, "smooth_sigma": 1.0,
    "clusters": 4, "min_motion": 0.5, "kmeans_iterations": 20, "kmeans_samples": 4096,
    "density_mode": "none", "top_fraction": 0.01, "normalization_quantile": 0.99,
    "normalization_samples_per_frame": 128, "normalization_max_samples": 50000,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    # All scientific outputs are exclusive-create. Failed stages leave partial outputs
    # for inspection; a new path is required for retrying.
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_hashes():
    root = Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


def environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ["numpy", "opencv-python", "opencv-python-headless", "torch", "torchvision"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "command": sys.argv, "source_sha256": source_hashes()}


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or supplied.keys() - config.keys():
            raise ValueError("Configuration must be an object with known keys only")
        config.update(supplied)
    for name in ["seed", "height", "width", "updates", "clusters", "kmeans_iterations",
                 "kmeans_samples", "normalization_samples_per_frame", "normalization_max_samples"]:
        if type(config[name]) is not int or config[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid integer configuration: {name}")
    if min(config["height"], config["width"]) < 128:
        raise ValueError("height/width must be >=128 for RAFT-large")
    for name in ["smooth_sigma", "min_motion", "top_fraction", "normalization_quantile"]:
        if isinstance(config[name], bool) or not isinstance(config[name], (int, float)) or not np.isfinite(config[name]):
            raise ValueError(f"Invalid finite configuration: {name}")
    if config["smooth_sigma"] < 0 or config["min_motion"] < 0:
        raise ValueError("Smoothing/motion gate must be nonnegative")
    if not 0 < config["top_fraction"] <= 1 or not 0 < config["normalization_quantile"] <= 1:
        raise ValueError("Quantile and top fraction must be in (0,1]")
    if not isinstance(config["weights"], str) or config["weights"] not in {"C_T_V1", "C_T_V2", "C_T_SKHT_V1", "C_T_SKHT_V2"}:
        raise ValueError("Use an explicit supported RAFT-large weight enum, not DEFAULT")
    if not isinstance(config["device"], str) or config["device"] not in {"cuda", "cpu"} or type(config["mixed_precision"]) is not bool:
        raise ValueError("device must be cuda/cpu; mixed_precision must be boolean")
    if config["mixed_precision"] and config["device"] != "cuda":
        raise ValueError("Mixed precision requires CUDA")
    if not isinstance(config["density_mode"], str) or config["density_mode"] not in {"none", "external"}:
        raise ValueError("density_mode must be none or external")
    return config


def load_manifest(path, config):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != 6 or set(reader.fieldnames) != {"id", "split", "group", "frames", "labels", "density"}:
            raise ValueError("CSV columns must be id,split,group,frames,labels,density")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    ids, directories, groups = set(), set(), {}
    records = []
    for row in rows:
        if any(value is None for value in row.values()) or None in row:
            raise ValueError("Malformed CSV row")
        row = {k: v.strip() for k, v in row.items()}
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row["id"]) or row["id"] in ids:
            raise ValueError("Video ids must be unique safe names")
        ids.add(row["id"])
        if row["split"] not in {"train", "val", "test"} or not row["group"]:
            raise ValueError("Each row requires train/val/test and a source group")
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Source group crosses splits")
        if not row["frames"]:
            raise ValueError("Frame directory is required")
        directory = (path.parent / row["frames"]).resolve()
        if directory in directories:
            raise ValueError("Frame directory reused across rows")
        directories.add(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing frame directory: {directory}")
        frames = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if len(frames) < 2 or len({p.stem for p in frames}) != len(frames):
            raise ValueError("Need >=2 images with unique stems, sorted in chronological filename order")
        label_path = (path.parent / row["labels"]).resolve() if row["labels"] else None
        if label_path is None:
            if row["split"] != "train":
                raise ValueError("Validation/test rows need per-frame binary .npy labels")
            labels = np.zeros(len(frames), dtype=np.uint8)
        else:
            labels = np.load(label_path, allow_pickle=False)
        if labels.shape != (len(frames),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must be a binary vector with one entry per input frame")
        if row["split"] == "train" and np.any(labels):
            raise ValueError("Calibration training data must be normal-only")
        density = (path.parent / row["density"]).resolve() if row["density"] else None
        if config["density_mode"] == "external" and (density is None or not density.is_dir()):
            raise ValueError("External density mode requires a density directory for every video")
        if config["density_mode"] == "none" and density is not None:
            raise ValueError("Density paths supplied but density_mode=none; resolve configuration explicitly")
        records.append({"id": row["id"], "split": row["split"], "group": row["group"],
                        "frames": frames, "labels": labels.astype(np.uint8),
                        "label_path": label_path, "density": density})
    if {r["split"] for r in records} != {"train", "val", "test"}:
        raise ValueError("Manifest must include train, val and test")
    return records
