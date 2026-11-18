"""Synthetic integration checks; these outputs are not scientific results."""

import csv
import json

import numpy as np
import pytest

from raftcrowd.io import DEFAULT_CONFIG, sha256
from raftcrowd.metrics import select_threshold
from raftcrowd.pipeline import (
    calibrate,
    evaluate_cache,
    load_cache,
    prepare_cache,
    score_map,
)
from raftcrowd.visualize import visualize


class SyntheticEstimator:
    metadata = {"backend": "SYNTHETIC_TEST_ONLY", "purpose": "pipeline integration test"}

    def __call__(self, first, second):
        assert first.shape == second.shape == (128, 128, 3)
        y, x = np.mgrid[:128, :128] / 127.0
        amplitude = float(second[0, 0, 0]) / 255.0 + 0.5
        flow = np.stack((amplitude * (2.0 - 0.5 * x), amplitude * (1.0 - 0.25 * y)), axis=-1)
        return flow.astype(np.float32), 0.001


@pytest.fixture
def synthetic_cache(tmp_path):
    cv2 = pytest.importorskip("cv2")
    config = dict(DEFAULT_CONFIG)
    config.update(
        device="cpu", height=128, width=128, smooth_sigma=0.0,
        clusters=2, min_motion=0.0, kmeans_samples=32,
        normalization_samples_per_frame=32, normalization_max_samples=64,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split", "group", "frames", "labels", "density"])
        writer.writeheader()
        for split_index, split in enumerate(["train", "val", "test"]):
            directory = tmp_path / split
            directory.mkdir()
            for frame_index in range(3):
                brightness = 10 + 30 * split_index + 10 * frame_index
                frame = np.full((128, 128, 3), brightness, dtype=np.uint8)
                assert cv2.imwrite(str(directory / f"frame_{frame_index:04d}.png"), frame)
            label_path = tmp_path / f"{split}_labels.npy"
            labels = [0, 0, 0] if split == "train" else [0, 0, 1]
            np.save(label_path, np.array(labels, dtype=np.uint8), allow_pickle=False)
            writer.writerow({
                "id": f"{split}_video", "split": split, "group": f"{split}_source",
                "frames": directory.name, "labels": label_path.name, "density": "",
            })
    cache = tmp_path / "cache"
    meta = prepare_cache(manifest_path, config_path, cache, estimator=SyntheticEstimator())
    return {"root": tmp_path, "cache": cache, "meta": meta,
            "manifest": manifest_path, "config": config_path}


def test_extract_calibrate_evaluate_preserves_frozen_protocol(synthetic_cache):
    setup = synthetic_cache
    cache, root = setup["cache"], setup["root"]
    assert setup["meta"]["estimator"]["backend"] == "SYNTHETIC_TEST_ONLY"
    assert (cache / "complete.json").is_file()
    assert [p["index"] for p in setup["meta"]["videos"][0]["pairs"]] == [1, 2]
    model_path = root / "model.json"
    model = calibrate(cache, model_path)
    assert model["threshold_source"] == "validation_labels_only"
    assert model["normalization"]["training_pairs"] == 2
    assert set(model["thresholds"]) == set(model["methods"])
    assert "pressure_only" not in model["methods"]
    assert len(model["validation_predictions"]) == 2
    assert all(row["video"] == "val_video" for row in model["validation_predictions"])
    for method, threshold in model["thresholds"].items():
        validation = model["validation_predictions"]
        assert threshold == select_threshold(
            [row["label"] for row in validation], [row[method] for row in validation]
        )
    report = evaluate_cache(cache, model_path, root / "evaluation")
    assert report["aggregation"] == "pooled_scored_frames"
    assert "first frame excluded" in report["frame_alignment"]
    assert set(report["methods"]) == set(model["methods"])
    for method, results in report["methods"].items():
        pooled = results["pooled"]
        assert results["threshold"] == model["thresholds"][method]
        assert sum(pooled[key] for key in ["tp", "tn", "fp", "fn"]) == 2
        assert 0 <= pooled["roc_auc"] <= 1
        assert 0 <= pooled["eer"] <= 1
        assert set(results["per_video"]) == {"test_video"}
    evaluation = root / "evaluation"
    with (evaluation / "frame_scores.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert [row["index"] for row in predictions] == ["1", "2"]
    assert [row["label"] for row in predictions] == ["0", "1"]
    receipt = json.loads((evaluation / "complete.json").read_text(encoding="utf-8"))
    assert receipt["metrics_sha256"] == sha256(evaluation / "metrics.json")
    assert receipt["frame_scores_sha256"] == sha256(evaluation / "frame_scores.csv")


def test_calibration_does_not_consult_test_labels(synthetic_cache):
    setup = synthetic_cache
    original = calibrate(setup["cache"], setup["root"] / "original_model.json")
    # Simulate a separately versioned input annotation set: both source and
    # cached copies and their receipts are updated. Train/val inputs are fixed.
    changed_labels = np.array([0, 1, 0], dtype=np.uint8)
    source_labels = setup["root"] / "test_labels.npy"
    cached_labels = setup["cache"] / "test_video" / "labels.npy"
    np.save(source_labels, changed_labels, allow_pickle=False)
    np.save(cached_labels, changed_labels, allow_pickle=False)
    cache_json = setup["cache"] / "cache.json"
    meta = json.loads(cache_json.read_text(encoding="utf-8"))
    test_record = next(v for v in meta["videos"] if v["split"] == "test")
    test_record["labels_sha256"] = sha256(cached_labels)
    test_record["label_source_sha256"] = sha256(source_labels)
    cache_json.write_text(json.dumps(meta), encoding="utf-8")
    (setup["cache"] / "complete.json").write_text(
        json.dumps({"cache_sha256": sha256(cache_json)}), encoding="utf-8"
    )
    changed = calibrate(setup["cache"], setup["root"] / "changed_model.json")
    assert changed["cache_sha256"] != original["cache_sha256"]
    for key in ["thresholds", "scales", "normalization", "validation_predictions"]:
        assert changed[key] == original[key]


def test_corrupt_training_pair_is_rejected(synthetic_cache):
    setup = synthetic_cache
    file = setup["cache"] / "train_video" / "pair_000001.npz"
    file.write_bytes(file.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Cached pair checksum mismatch"):
        calibrate(setup["cache"], setup["root"] / "model.json")
    assert not (setup["root"] / "model.json").exists()


def test_corrupt_test_pair_is_rejected_at_evaluation(synthetic_cache):
    setup = synthetic_cache
    model_path = setup["root"] / "model.json"
    calibrate(setup["cache"], model_path)
    file = setup["cache"] / "test_video" / "pair_000002.npz"
    file.write_bytes(file.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Cached pair checksum mismatch"):
        evaluate_cache(setup["cache"], model_path, setup["root"] / "evaluation")
    assert not (setup["root"] / "evaluation" / "complete.json").exists()


def test_cache_metadata_receipt_is_checked(synthetic_cache):
    setup = synthetic_cache
    file = setup["cache"] / "cache.json"
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Cache metadata checksum mismatch"):
        load_cache(setup["cache"])


def test_existing_scientific_outputs_are_not_overwritten(synthetic_cache):
    setup = synthetic_cache
    model_path = setup["root"] / "model.json"
    calibrate(setup["cache"], model_path)
    model_hash = sha256(model_path)
    with pytest.raises(FileExistsError):
        calibrate(setup["cache"], model_path)
    assert sha256(model_path) == model_hash
    output = setup["root"] / "evaluation"
    evaluate_cache(setup["cache"], model_path, output)
    report_hash = sha256(output / "metrics.json")
    with pytest.raises(FileExistsError):
        evaluate_cache(setup["cache"], model_path, output)
    assert sha256(output / "metrics.json") == report_hash
    with pytest.raises(FileExistsError):
        prepare_cache(setup["manifest"], setup["config"], setup["cache"], estimator=SyntheticEstimator())


def test_score_map_uses_frozen_scales_and_cue_mean():
    arrays = {
        "compression": np.array([[0.0, 1.0], [4.0, 8.0]]),
        "magnitude_variance": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "global_magnitude_variance": np.full((2, 2), 8.0),
    }
    scales = {"compression": 4.0, "magnitude_variance": 4.0,
              "global_magnitude_variance": 16.0}
    method = {"cues": ["compression", "magnitude_variance"], "segmented": True}
    result = score_map(arrays, method, scales)
    np.testing.assert_allclose(result, [[0.125, 0.375], [0.875, 1.0]])
    global_method = dict(method, segmented=False)
    global_result = score_map(arrays, global_method, scales)
    np.testing.assert_allclose(global_result, [[0.25, 0.375], [0.75, 0.75]])


def test_visualization_writes_six_panel_png_exclusively(synthetic_cache):
    cv2 = pytest.importorskip("cv2")
    setup = synthetic_cache
    model_path = setup["root"] / "model.json"
    calibrate(setup["cache"], model_path)
    output = setup["root"] / "six_panel.png"
    assert visualize(setup["cache"], model_path, "test_video", 1, output) == output
    decoded = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (3 * (128 + 32), 2 * 128, 3)
    assert np.any(decoded > 0)
    before = sha256(output)
    with pytest.raises(FileExistsError):
        visualize(setup["cache"], model_path, "test_video", 1, output)
    assert sha256(output) == before


def test_visualization_rejects_unscored_first_frame(synthetic_cache):
    setup = synthetic_cache
    model_path = setup["root"] / "model.json"
    calibrate(setup["cache"], model_path)
    output = setup["root"] / "invalid_frame.png"
    with pytest.raises(ValueError, match="scored second-frame index"):
        visualize(setup["cache"], model_path, "test_video", 0, output)
    assert not output.exists()


def test_visualization_rejects_changed_source_frame(synthetic_cache):
    setup = synthetic_cache
    model_path = setup["root"] / "model.json"
    calibrate(setup["cache"], model_path)
    source = setup["root"] / "test" / "frame_0000.png"
    source.write_bytes(source.read_bytes() + b"changed source identity")
    output = setup["root"] / "changed_source.png"
    with pytest.raises(ValueError, match="Source image identity changed"):
        visualize(setup["cache"], model_path, "test_video", 1, output)
    assert not output.exists()


def _enable_external_density(setup):
    config = json.loads(setup["config"].read_text(encoding="utf-8"))
    config["density_mode"] = "external"
    setup["config"].write_text(json.dumps(config), encoding="utf-8")
    with setup["manifest"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    for row in rows:
        directory = setup["root"] / f"{row['split']}_density"
        directory.mkdir()
        # Only source grids for the two evaluated pairs are supplied. There is
        # intentionally no density for the final frame, which has no outgoing flow.
        np.save(directory / "frame_0000.npy", np.full((128, 128), 2.0), allow_pickle=False)
        np.save(directory / "frame_0001.npy", np.full((128, 128), 7.0), allow_pickle=False)
        row["density"] = directory.name
    with setup["manifest"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_external_density_uses_source_frame_grid_and_enables_pressure_variants(synthetic_cache):
    setup = synthetic_cache
    _enable_external_density(setup)
    cache = setup["root"] / "external_density_cache"
    meta = prepare_cache(setup["manifest"], setup["config"], cache, estimator=SyntheticEstimator())
    for video in meta["videos"]:
        for pair in video["pairs"]:
            index = pair["index"]
            expected_density = 2.0 if index == 1 else 7.0
            source = setup["root"] / f"{video['split']}_density" / f"frame_{index - 1:04d}.npy"
            assert pair["source_density_sha256"] == sha256(source)
            with np.load(cache / video["id"] / pair["file"], allow_pickle=False) as arrays:
                magnitude = np.hypot(arrays["flow"][..., 0], arrays["flow"][..., 1])
                np.testing.assert_allclose(arrays["pressure"], expected_density * magnitude, rtol=1e-6)
    model_path = setup["root"] / "external_model.json"
    model = calibrate(cache, model_path)
    assert "pressure" in model["methods"]["full"]["cues"]
    assert model["methods"]["pressure_only"]["cues"] == ["pressure"]
    assert "pressure" not in model["methods"]["without_pressure"]["cues"]
    assert model["scales"]["pressure"] > 0
    report = evaluate_cache(cache, model_path, setup["root"] / "external_evaluation")
    assert {"pressure_only", "without_pressure"} <= set(report["methods"])


def test_external_density_rejects_automatic_grid_resizing(synthetic_cache):
    setup = synthetic_cache
    _enable_external_density(setup)
    density = setup["root"] / "train_density" / "frame_0000.npy"
    np.save(density, np.ones((127, 128)), allow_pickle=False)
    cache = setup["root"] / "bad_density_cache"
    with pytest.raises(ValueError, match="aligned to the resized source-frame grid"):
        prepare_cache(setup["manifest"], setup["config"], cache, estimator=SyntheticEstimator())
    assert not (cache / "complete.json").exists()
