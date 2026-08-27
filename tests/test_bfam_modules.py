"""Shape and numerical checks for BFAM-Net modules."""

import pytest
import torch

from ultralytics.nn.modules import BCFM, BFRE, CSCG, RFAM
from ultralytics.nn.modules.bfam import _DynamicUpsample


@pytest.mark.parametrize("module", [BFRE(16, 24), BCFM(16, 24), RFAM(16, 24, 0.25)])
def test_single_input_modules(module: torch.nn.Module) -> None:
    module.eval()
    inputs = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        outputs = module(inputs)
    assert outputs.shape == (1, 24, 8, 8)
    assert torch.isfinite(outputs).all()


def test_cscg_fuses_two_scales() -> None:
    module = CSCG(low_channels=16, high_channels=8, out_channels=8, scale_factor=2).eval()
    low_resolution = torch.randn(1, 16, 4, 4)
    high_resolution = torch.randn(1, 8, 8, 8)
    with torch.no_grad():
        outputs = module(low_resolution, high_resolution)
    assert outputs.shape == high_resolution.shape
    assert torch.isfinite(outputs).all()


def test_dynamic_upsample_uses_grouped_subpixel_offsets() -> None:
    module = _DynamicUpsample(channels=16, scale_factor=2, groups=4).eval()
    assert module.offset[-1].out_channels == 2 * 4 * 2 * 2
    inputs = torch.randn(1, 16, 4, 5)
    with torch.no_grad():
        outputs = module(inputs)
    assert outputs.shape == (1, 16, 8, 10)
    assert torch.isfinite(outputs).all()
