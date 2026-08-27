"""Train BFAM-Net with the manuscript training defaults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DEFAULT_MODEL = ROOT / "configs" / "bfam-net.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="BFAM-Net model YAML")
    parser.add_argument("--pretrained", type=Path, help="Optional compatible checkpoint for partial weight transfer")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="CUDA device, 'cpu', or 'mps'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "bfam-net")
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    if args.pretrained:
        model.load(str(args.pretrained))

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        val=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
