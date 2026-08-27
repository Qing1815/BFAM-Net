"""Build BFAM-Net and execute an end-to-end random-tensor forward pass."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

MODEL_CONFIG = ROOT / "configs" / "bfam-net.yaml"


def test_bfam_net_smoke() -> None:
    model = YOLO(str(MODEL_CONFIG), verbose=False).model.eval()
    module_names = {layer.__class__.__name__ for layer in model.modules()}
    assert {"BFRE", "BCFM", "RFAM", "CSCG"}.issubset(module_names)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 4_500_000 < parameter_count < 5_000_000
    assert model.stride.tolist() == [8.0, 16.0, 32.0]

    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None


if __name__ == "__main__":
    test_bfam_net_smoke()
    print("BFAM-Net smoke test passed.")
