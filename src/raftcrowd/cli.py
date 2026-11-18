"""Command-line workflow; importing help never imports PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import sha256, write_json


def video_frames(video, output):
    import cv2
    video = Path(video).resolve()
    original_hash = sha256(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {video}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    index, timestamps = 0, []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            filename = output / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(filename), image):
                raise RuntimeError(f"Could not write frame {filename}")
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            index += 1
    finally:
        capture.release()
    if index < 2:
        raise ValueError("Video yielded fewer than two frames")
    if sha256(video) != original_hash:
        raise ValueError("Video changed during decoding")
    write_json(output / "decode.json", {"video": str(video), "sha256": original_hash,
                                      "frames": index, "nominal_fps": fps,
                                      "decoder_timestamps_ms_unverified": timestamps,
                                      "note": "OpenCV timestamps are diagnostic; no native-PTS alignment claim"})
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prospective RAFT-CrowdFlow implementation; no recovered paper results")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Frozen RAFT flow and numeric cues for all manifest rows")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--config", required=True)
    extract.add_argument("--out", required=True)
    fit = sub.add_parser("calibrate", help="Train-only scales and validation-only thresholds")
    fit.add_argument("--cache", required=True)
    fit.add_argument("--out", required=True)
    evaluation = sub.add_parser("evaluate", help="Fixed-threshold test evaluation and all declared ablations")
    evaluation.add_argument("--cache", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--out", required=True)
    panels = sub.add_parser("visualize", help="Six panels from a single cached flow")
    panels.add_argument("--cache", required=True)
    panels.add_argument("--model", required=True)
    panels.add_argument("--video-id", required=True)
    panels.add_argument("--frame-index", required=True, type=int)
    panels.add_argument("--out", required=True)
    frames = sub.add_parser("video-frames", help="Extract every decoded video frame to a new directory")
    frames.add_argument("--video", required=True)
    frames.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "video-frames":
        print("Decoded frames:", video_frames(args.video, args.out))
        return
    from .pipeline import calibrate, evaluate_cache, prepare_cache
    if args.command == "extract":
        prepare_cache(args.manifest, args.config, args.out)
    elif args.command == "calibrate":
        model = calibrate(args.cache, args.out)
        print(json.dumps({"thresholds": model["thresholds"], "normalization": model["normalization"]}, indent=2))
    elif args.command == "evaluate":
        report = evaluate_cache(args.cache, args.model, args.out)
        print(json.dumps({k: v["pooled"] for k, v in report["methods"].items()}, indent=2))
    else:
        from .visualize import visualize
        print(visualize(args.cache, args.model, args.video_id, args.frame_index, args.out))


if __name__ == "__main__":
    main()
