"""Run BFAM-Net inference on an image, directory, video, or stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("runs") / "predict")
    parser.add_argument("--name", default="bfam-net")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO(str(args.weights)).predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
