"""Six panels derived from one checked cached flow, never relabeled as safety truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import sha256
from .pipeline import load_cache, load_model, read_rgb, score_map


def visualize(cache, model_path, video_id, frame_index, output):
    import cv2
    root, meta, cache_hash = load_cache(cache)
    model = load_model(model_path, cache_hash, meta)
    video = next((v for v in meta["videos"] if v["id"] == video_id), None)
    if video is None:
        raise ValueError("Unknown video id")
    pair = next((p for p in video["pairs"] if p["index"] == frame_index), None)
    if pair is None:
        raise ValueError("frame-index must be a scored second-frame index, from 1 to N-1")
    source = video["frames"][frame_index - 1]
    if sha256(source) != video["frame_sha256"][frame_index - 1]:
        raise ValueError("Source image identity changed")
    file = root / video_id / pair["file"]
    if sha256(file) != pair["sha256"]:
        raise ValueError("Cached pair changed")
    with np.load(file, allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}
    flow = arrays["flow"]
    mag = np.hypot(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = ((np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) * 180 / (2 * np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (255 * np.clip(mag / model["scales"]["magnitude"], 0, 1)).astype(np.uint8)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def heat(values):
        return cv2.applyColorMap((255 * np.clip(values, 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)

    full = score_map(arrays, model["methods"]["full"], model["scales"])
    segments = arrays["segments"]
    palette = np.array([[255, 180, 30], [20, 210, 240], [190, 50, 220], [70, 220, 90], [80, 80, 250]], np.uint8)
    grouped = np.zeros((*segments.shape, 3), np.uint8)
    active = segments >= 0
    grouped[active] = palette[segments[active] % len(palette)]
    panels = [cv2.cvtColor(read_rgb(source, meta["config"]), cv2.COLOR_RGB2BGR), flow_color,
              heat(full), heat(arrays["magnitude"] / model["scales"]["magnitude"]),
              heat(arrays["compression"] / model["scales"]["compression"]), grouped]
    titles = ["Source RGB (t)", "Flow direction / magnitude", "CFI motion score [0,1]",
              "Magnitude / training scale", "Compression / training scale", "Motion groups (not people)"]
    labeled = []
    for panel, title in zip(panels, titles):
        bar = np.zeros((32, panel.shape[1], 3), np.uint8)
        text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
        font_scale = min(0.5, (panel.shape[1] - 12) / max(text_width, 1))
        cv2.putText(bar, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(np.vstack((bar, panel)))
    mosaic = np.vstack([np.hstack(labeled[i:i + 2]) for i in [0, 2, 4]])
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Visualization output must be a PNG path")
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mosaic)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    with output.open("xb") as handle:
        handle.write(encoded.tobytes())
    return output
