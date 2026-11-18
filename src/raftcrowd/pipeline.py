"""Extract once, calibrate on train/validation, evaluate fixed methods on test."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .features import CueConfig, aggregate_map, compute_cues
from .io import environment, identity, load_config, load_manifest, sha256, source_hashes, write_json
from .metrics import evaluate, fixed_metrics, select_threshold


def cue_config(config):
    return CueConfig(**{k: config[k] for k in ["clusters", "min_motion", "kmeans_iterations", "kmeans_samples", "seed"]})


def methods(config):
    cues = ["compression", "direction_variance", "magnitude_variance"]
    if config["density_mode"] == "external":
        cues.append("pressure")
    result = {"full": {"cues": cues, "segmented": True},
              "magnitude_only": {"cues": ["magnitude"], "segmented": True},
              "no_segmentation": {"cues": cues, "segmented": False}}
    for cue in cues:
        result[f"{cue}_only"] = {"cues": [cue], "segmented": True}
        result[f"without_{cue}"] = {"cues": [c for c in cues if c != cue], "segmented": True}
    return result


def read_rgb(path, config):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode frame {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (config["width"], config["height"]), interpolation=cv2.INTER_LINEAR)


def prepare_cache(manifest_path, config_path, output, estimator=None):
    """An injected estimator is only for synthetic tests, and is marked in receipts."""
    config = load_config(config_path)
    records = load_manifest(manifest_path, config)
    # Hash all originals before inference. Whole-video copies cannot masquerade as
    # different splits. Group correctness still requires a human dataset audit.
    seen_videos = {}
    for record in records:
        record["frame_sha256"] = [sha256(p) for p in record["frames"]]
        video_identity = identity(record["frame_sha256"])
        if video_identity in seen_videos:
            raise ValueError("Identical frame sequence appears in multiple manifest rows")
        seen_videos[video_identity] = record["id"]
        record["density_sha256"] = []
        if record["density"] is not None:
            # Cheap coverage/shape audit before constructing a GPU model.
            for frame in record["frames"][:-1]:
                density_path = record["density"] / (frame.stem + ".npy")
                density = np.load(density_path, allow_pickle=False, mmap_mode="r")
                if density.shape != (config["height"], config["width"]) or not np.isfinite(density).all() or np.any(density < 0):
                    raise ValueError("External density must be finite, nonnegative and aligned to the resized source-frame grid")
                record["density_sha256"].append(sha256(density_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if estimator is None:
        from .raft import RaftEstimator
        estimator = RaftEstimator(config)
    backend = getattr(estimator, "metadata", {"backend": "injected_unverified_test_estimator"})
    meta = {"format": 1, "config": config, "manifest_sha256": sha256(manifest_path),
            "environment": environment(), "estimator": backend, "videos": []}
    import cv2
    for record in records:
        video_dir = output / record["id"]
        video_dir.mkdir()
        np.save(video_dir / "labels.npy", record["labels"], allow_pickle=False)
        video = {"id": record["id"], "split": record["split"], "group": record["group"],
                 "frames": [str(p) for p in record["frames"]],
                 "frame_sha256": record["frame_sha256"],
                 "label_source_sha256": sha256(record["label_path"]) if record["label_path"] else None,
                 "labels_sha256": sha256(video_dir / "labels.npy"), "pairs": []}
        first = read_rgb(record["frames"][0], config)
        for index, frame in enumerate(record["frames"][1:], 1):
            # Recheck the input hash as it is read; reject concurrent source changes.
            if sha256(record["frames"][index - 1]) != record["frame_sha256"][index - 1] or sha256(frame) != record["frame_sha256"][index]:
                raise ValueError("Input frame changed during extraction")
            start = time.perf_counter()
            second = read_rgb(frame, config)
            flow, inference_seconds = estimator(first, second)
            flow = np.asarray(flow, dtype=np.float32)
            if flow.shape != (config["height"], config["width"], 2):
                raise ValueError("Estimator output must match the unpadded resized image dimensions")
            if config["smooth_sigma"] > 0:
                flow = cv2.GaussianBlur(flow, (0, 0), config["smooth_sigma"])
            density, density_hash = None, None
            if record["density"] is not None:
                # Optical flow is defined on the first image's pixel coordinates.
                density_path = record["density"] / (record["frames"][index - 1].stem + ".npy")
                density_hash = sha256(density_path)
                if density_hash != record["density_sha256"][index - 1]:
                    raise ValueError("External density changed during extraction")
                density = np.load(density_path, allow_pickle=False)
                if density.shape != flow.shape[:2]:
                    raise ValueError("External density must be aligned to the resized source-frame grid; automatic resizing is forbidden")
            cues = compute_cues(flow, cue_config(config), density=density)
            global_cues = compute_cues(flow, cue_config(config), density=density, segment=False)
            arrays = {name: value.astype(np.float32) for name, value in cues.items()
                      if name not in {"mask", "segments"}}
            arrays.update(mask=cues["mask"], segments=cues["segments"].astype(np.int32), flow=flow,
                          global_direction_variance=global_cues["direction_variance"].astype(np.float32),
                          global_magnitude_variance=global_cues["magnitude_variance"].astype(np.float32))
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Cached cue/flow exceeds finite float32 range")
            pair_name = f"pair_{index:06d}.npz"
            np.savez_compressed(video_dir / pair_name, **arrays)
            video["pairs"].append({"index": index, "file": pair_name,
                                   "sha256": sha256(video_dir / pair_name),
                                   "source_density_sha256": density_hash,
                                   "inference_seconds": float(inference_seconds),
                                   "extraction_seconds": time.perf_counter() - start})
            first = second
        meta["videos"].append(video)
        print(f"Extracted {record['id']}: {len(video['pairs'])} pairs", flush=True)
    # Receipt last. A partial directory cannot be calibrated or evaluated.
    write_json(output / "cache.json", meta)
    write_json(output / "complete.json", {"cache_sha256": sha256(output / "cache.json")})
    return meta


def load_cache(root):
    root = Path(root)
    receipt = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if sha256(root / "cache.json") != receipt["cache_sha256"]:
        raise ValueError("Cache metadata checksum mismatch")
    meta = json.loads((root / "cache.json").read_text(encoding="utf-8"))
    if meta["environment"]["source_sha256"] != source_hashes():
        raise ValueError("Implementation changed since extraction; preserve old code or create a new cache")
    return root, meta, receipt["cache_sha256"]


def iter_pairs(root, meta, split=None):
    for video in meta["videos"]:
        if split is not None and video["split"] != split:
            continue
        label_path = root / video["id"] / "labels.npy"
        if sha256(label_path) != video["labels_sha256"]:
            raise ValueError("Cached labels changed")
        labels = np.load(label_path, allow_pickle=False)
        for pair in video["pairs"]:
            file = root / video["id"] / pair["file"]
            if sha256(file) != pair["sha256"]:
                raise ValueError(f"Cached pair checksum mismatch: {file}")
            with np.load(file, allow_pickle=False) as shard:
                arrays = {key: shard[key] for key in shard.files}
            yield video, pair, int(labels[pair["index"]]), arrays


def score_map(arrays, method, scales):
    parts = []
    for cue in method["cues"]:
        key = "global_" + cue if not method["segmented"] and cue in {"direction_variance", "magnitude_variance"} else cue
        values = np.asarray(arrays[key], dtype=np.float64)
        parts.append(np.clip(values / scales[key], 0, 1))
    return np.mean(parts, axis=0)  # equal weights, sum=1; never fitted on test


def score_split(root, meta, model, split):
    rows = []
    for video, pair, label, arrays in iter_pairs(root, meta, split):
        row = {"video": video["id"], "index": pair["index"], "label": label}
        for name, method in model["methods"].items():
            row[name] = aggregate_map(score_map(arrays, method, model["scales"]), meta["config"]["top_fraction"])
        rows.append(row)
    return rows


def calibrate(cache, output):
    root, meta, cache_hash = load_cache(cache)
    config = meta["config"]
    definitions = methods(config)
    cues = sorted({("global_" + cue if not method["segmented"] and cue in
                    {"direction_variance", "magnitude_variance"} else cue)
                   for method in definitions.values() for cue in method["cues"]})
    # Bounded random-priority reservoirs are deterministic for a fixed manifest
    # traversal order. Only normal training frames enter these reservoirs.
    rng = np.random.default_rng(config["seed"])
    samples = {cue: np.empty(0) for cue in cues}
    priorities = {cue: np.empty(0) for cue in cues}
    training_pairs = 0
    for _, _, label, arrays in iter_pairs(root, meta, "train"):
        if label != 0:
            raise ValueError("Anomalous training label in cache")
        training_pairs += 1
        active = np.flatnonzero(arrays["mask"])
        if not len(active):
            continue
        indices = rng.choice(active, min(len(active), config["normalization_samples_per_frame"]), replace=False)
        for cue in cues:
            samples[cue] = np.concatenate((samples[cue], arrays[cue].ravel()[indices]))
            priorities[cue] = np.concatenate((priorities[cue], rng.random(len(indices))))
            cap = config["normalization_max_samples"]
            if len(samples[cue]) > cap:
                keep = np.argpartition(priorities[cue], -cap)[-cap:]
                samples[cue], priorities[cue] = samples[cue][keep], priorities[cue][keep]
    if not training_pairs:
        raise ValueError("No training pairs")
    scales = {}
    zero_scale_cues = []
    for cue in cues:
        value = float(np.quantile(samples[cue], config["normalization_quantile"])) if len(samples[cue]) else 0.0
        if value <= 1e-12:
            zero_scale_cues.append(cue)
            value = 1.0  # documented fallback when training cue has no positive range
        scales[cue] = value
    model = {"format": 1, "cache_sha256": cache_hash, "methods": definitions, "scales": scales,
             "normalization": {"training_pairs": training_pairs, "quantile": config["normalization_quantile"],
                               "sample_counts": {k: len(v) for k, v in samples.items()},
                               "zero_scale_fallback_cues": zero_scale_cues},
             "threshold_source": "validation_labels_only", "thresholds": {}, "environment": environment()}
    validation = score_split(root, meta, model, "val")
    labels = np.array([r["label"] for r in validation])
    for name in definitions:
        model["thresholds"][name] = select_threshold(labels, np.array([r[name] for r in validation]))
    model["validation_predictions"] = validation
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, model)
    return model


def load_model(path, cache_hash, meta):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model["cache_sha256"] != cache_hash:
        raise ValueError("Calibrated model belongs to a different cache")
    if model["methods"] != methods(meta["config"]):
        raise ValueError("Method definitions changed")
    if any(not math.isfinite(v) or v <= 0 for v in model["scales"].values()):
        raise ValueError("Invalid normalization scales")
    if set(model["thresholds"]) != set(model["methods"]) or not all(math.isfinite(v) for v in model["thresholds"].values()):
        raise ValueError("Invalid frozen thresholds")
    return model


def evaluate_cache(cache, model_path, output):
    root, meta, cache_hash = load_cache(cache)
    model = load_model(model_path, cache_hash, meta)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = score_split(root, meta, model, "test")
    labels = np.array([r["label"] for r in rows])
    report = {"status": "computed_from_supplied_inputs_not_historical_paper_results",
              "cache_sha256": cache_hash, "model_sha256": sha256(model_path), "aggregation": "pooled_scored_frames",
              "frame_alignment": "pair(t,t+1) predicts anomaly label[t+1]; first frame excluded",
              "environment": environment(), "methods": {}}
    for name in model["methods"]:
        scores = np.array([r[name] for r in rows])
        metrics = evaluate(labels, scores, model["thresholds"][name])
        per_video = {}
        for video_id in sorted({r["video"] for r in rows}):
            chosen = [r for r in rows if r["video"] == video_id]
            y = np.array([r["label"] for r in chosen])
            s = np.array([r[name] for r in chosen])
            if len(np.unique(y)) == 2:
                per_video[video_id] = evaluate(y, s, model["thresholds"][name])
            else:
                per_video[video_id] = dict(fixed_metrics(y, s, model["thresholds"][name]),
                                          roc_auc=None, eer=None, reason="single_class_video")
        report["methods"][name] = {"threshold": model["thresholds"][name], "pooled": metrics, "per_video": per_video}
    with (output / "frame_scores.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "index", "label", *model["methods"]])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "metrics.json", report)
    write_json(output / "complete.json", {"metrics_sha256": sha256(output / "metrics.json"),
                                         "frame_scores_sha256": sha256(output / "frame_scores.csv")})
    return report
