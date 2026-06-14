#!/usr/bin/env python3
"""Run ImageAI object detection on videos in ImageAI-master/data-videos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from importlib import metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGEAI_ROOT = REPO_ROOT / "ImageAI-master"
DEFAULT_VIDEO_DIR = IMAGEAI_ROOT / "data-videos"
DEFAULT_OUTPUT_DIR = IMAGEAI_ROOT / "output-videos"
DEFAULT_MPLCONFIGDIR = REPO_ROOT / ".matplotlib-cache"
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ImageAI video object detection. Output files are written as .avi "
            "because ImageAI appends that extension internally."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a YOLOv3/TinyYOLOv3 .h5 model file.",
    )
    parser.add_argument(
        "--model-type",
        choices=["yolo", "tiny-yolo"],
        default="tiny-yolo",
        help="Model architecture for the .h5 file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Video file or directory of videos to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for detected output videos.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for output video.",
    )
    parser.add_argument(
        "--min-probability",
        type=int,
        default=50,
        help="Minimum detection probability percentage.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Run detection every N frames.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional max seconds of video to process.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def iter_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def load_detector(model_path: Path, model_type: str):
    sys.path.insert(0, str(IMAGEAI_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
    try:
        keras_version = metadata.version("keras")
    except metadata.PackageNotFoundError:
        keras_version = ""
    if keras_version.startswith("3."):
        raise SystemExit(
            "This vendored ImageAI 2.1.5 code is not compatible with Keras 3. "
            f"Installed keras version: {keras_version}. Use the legacy "
            "TensorFlow/Keras environment documented in README.md, or migrate "
            "the video runner to a modern detector such as Ultralytics YOLO."
        )
    try:
        from imageai.Detection import VideoObjectDetection
    except ImportError as exc:
        raise SystemExit(
            "ImageAI failed to import. This is usually a missing dependency or "
            f"Keras/TensorFlow compatibility issue. Original error: {exc}"
        ) from exc

    detector = VideoObjectDetection()
    if model_type == "yolo":
        detector.setModelTypeAsYOLOv3()
    else:
        detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(str(model_path))
    detector.loadModel()
    return detector


def main() -> int:
    args = parse_args()
    model_path = resolve_path(args.model)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)

    if not model_path.is_file():
        raise SystemExit(f"model file not found: {model_path}")

    videos = iter_videos(input_path)
    if not videos:
        raise SystemExit(f"no videos found in: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detector = load_detector(model_path, args.model_type)

    for video in videos:
        output_stem = output_dir / f"{video.stem}_detected"
        print(f"Processing {video}")
        output_path = detector.detectObjectsFromVideo(
            input_file_path=str(video),
            output_file_path=str(output_stem),
            frames_per_second=args.fps,
            frame_detection_interval=args.frame_interval,
            minimum_percentage_probability=args.min_probability,
            detection_timeout=args.timeout,
            log_progress=True,
        )
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
