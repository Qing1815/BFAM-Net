"""Export trained BFAM-Net weights to a deployment format."""

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
    parser.add_argument("--format", default="onnx", help="onnx, engine, openvino, torchscript, etc.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = YOLO(str(args.weights)).export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
    )
    print(output)


if __name__ == "__main__":
    main()
