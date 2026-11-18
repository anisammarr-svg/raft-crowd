"""Manifest/protocol admission checks using tiny synthetic files, without decoding."""

import csv
import json

import numpy as np
import pytest

from raftcrowd.io import DEFAULT_CONFIG, load_config, load_manifest


@pytest.fixture
def manifest_data(tmp_path):
    rows = []
    for split in ["train", "val", "test"]:
        frames = tmp_path / f"{split}_frames"
        frames.mkdir()
        # Dummy bytes intentionally cannot be decoded: this tests admission only.
        for name in ["000.png", "001.png", "002.png"]:
            (frames / name).write_bytes(b"synthetic admission fixture")
        labels = tmp_path / f"{split}_labels.npy"
        np.save(labels, [0, 0, 0] if split == "train" else [0, 0, 1])
        rows.append({
            "id": f"video_{split}",
            "split": split,
            "group": f"source_{split}",
            "frames": frames.name,
            "labels": labels.name,
            "density": "",
        })
    return tmp_path, rows


def write_manifest(tmp_path, rows, columns=None):
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["id", "split", "group", "frames", "labels", "density"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_defaults_and_partial_configuration(tmp_path):
    defaults = load_config()
    assert defaults == DEFAULT_CONFIG
    defaults["clusters"] = 999
    assert load_config()["clusters"] != 999
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 17, "device": "cpu"}))
    result = load_config(path)
    assert result["seed"] == 17
    assert result["device"] == "cpu"
    assert result["height"] == DEFAULT_CONFIG["height"]


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 1},
        {"seed": True},
        {"seed": -1},
        {"height": 127},
        {"height": 360.0},
        {"updates": 0},
        {"clusters": False},
        {"kmeans_samples": 0},
        {"normalization_max_samples": 0},
        {"smooth_sigma": float("nan")},
        {"min_motion": float("inf")},
        {"top_fraction": True},
        {"top_fraction": 0},
        {"normalization_quantile": 1.1},
        {"density_mode": "motion_proxy"},
        {"weights": "DEFAULT"},
        {"device": "mps"},
        {"mixed_precision": 1},
        {"device": "cpu", "mixed_precision": True},
        {"density_mode": []},
        {"weights": {}},
        {"device": []},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, changes):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changes))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("value", [[], None, True, "config"])
def test_config_requires_object(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_config(path)


def test_manifest_resolves_paths_sorts_frames_and_loads_labels(manifest_data):
    root, rows = manifest_data
    result = load_manifest(write_manifest(root, rows), load_config())
    assert [record["split"] for record in result] == ["train", "val", "test"]
    assert [frame.name for frame in result[0]["frames"]] == ["000.png", "001.png", "002.png"]
    assert all(frame.is_absolute() for record in result for frame in record["frames"])
    np.testing.assert_array_equal(result[2]["labels"], [0, 0, 1])


def test_train_labels_are_optional_and_assumed_normal(manifest_data):
    root, rows = manifest_data
    rows[0]["labels"] = ""
    result = load_manifest(write_manifest(root, rows), load_config())
    assert result[0]["label_path"] is None
    np.testing.assert_array_equal(result[0]["labels"], [0, 0, 0])


@pytest.mark.parametrize("split_index", [1, 2])
def test_validation_and_test_labels_are_required(manifest_data, split_index):
    root, rows = manifest_data
    rows[split_index]["labels"] = ""
    with pytest.raises(ValueError, match="need per-frame"):
        load_manifest(write_manifest(root, rows), load_config())


def test_source_group_cannot_cross_splits(manifest_data):
    root, rows = manifest_data
    rows[1]["group"] = rows[0]["group"]
    with pytest.raises(ValueError, match="crosses splits"):
        load_manifest(write_manifest(root, rows), load_config())


def test_duplicate_resolved_frame_directories_rejected(manifest_data):
    root, rows = manifest_data
    rows[1]["frames"] = "./train_frames/../train_frames"
    with pytest.raises(ValueError, match="reused"):
        load_manifest(write_manifest(root, rows), load_config())


def test_anomalous_training_labels_rejected(manifest_data):
    root, rows = manifest_data
    np.save(root / rows[0]["labels"], [0, 1, 0])
    with pytest.raises(ValueError, match="normal-only"):
        load_manifest(write_manifest(root, rows), load_config())


@pytest.mark.parametrize("labels", [[0, 1], [[0, 0, 1]], [0, 2, 1], [0, np.nan, 1]])
def test_invalid_label_shape_or_values_rejected(manifest_data, labels):
    root, rows = manifest_data
    np.save(root / rows[1]["labels"], labels)
    with pytest.raises(ValueError, match="binary vector"):
        load_manifest(write_manifest(root, rows), load_config())


def test_density_paths_rejected_when_density_disabled(manifest_data):
    root, rows = manifest_data
    rows[0]["density"] = "density_train"
    with pytest.raises(ValueError, match="density_mode=none"):
        load_manifest(write_manifest(root, rows), load_config())


def test_external_density_requires_all_video_directories(manifest_data):
    root, rows = manifest_data
    config = load_config()
    config["density_mode"] = "external"
    with pytest.raises(ValueError, match="every video"):
        load_manifest(write_manifest(root, rows), config)
    for row in rows:
        row["density"] = f"density_{row['split']}"
        (root / row["density"]).mkdir()
    result = load_manifest(write_manifest(root, rows), config)
    assert all(record["density"].is_dir() for record in result)


@pytest.mark.parametrize("video_id", ["../escape", "unsafe/name", "", "-prefix", "a b"])
def test_unsafe_video_ids_rejected(manifest_data, video_id):
    root, rows = manifest_data
    rows[0]["id"] = video_id
    with pytest.raises(ValueError, match="safe names"):
        load_manifest(write_manifest(root, rows), load_config())


def test_duplicate_ids_rejected(manifest_data):
    root, rows = manifest_data
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="safe names"):
        load_manifest(write_manifest(root, rows), load_config())


def test_all_three_splits_required(manifest_data):
    root, rows = manifest_data
    with pytest.raises(ValueError, match="include train, val and test"):
        load_manifest(write_manifest(root, rows[:2]), load_config())


def test_duplicate_image_stems_rejected(manifest_data):
    root, rows = manifest_data
    (root / rows[0]["frames"] / "000.jpg").write_bytes(b"duplicate stem")
    with pytest.raises(ValueError, match="unique stems"):
        load_manifest(write_manifest(root, rows), load_config())


def test_image_suffix_directories_do_not_count_as_frames(manifest_data):
    root, rows = manifest_data
    rows[0]["labels"] = ""
    frames = root / rows[0]["frames"]
    (frames / "001.png").unlink()
    (frames / "002.png").unlink()
    (frames / "001.png").mkdir()
    with pytest.raises(ValueError, match=">=2 images"):
        load_manifest(write_manifest(root, rows), load_config())


def test_duplicate_csv_headers_rejected(manifest_data):
    root, rows = manifest_data
    path = write_manifest(root, rows, ["id", "split", "group", "frames", "labels", "density", "id"])
    with pytest.raises(ValueError, match="CSV columns"):
        load_manifest(path, load_config())
