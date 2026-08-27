"""Hierarchical residual feature alignment for BFAM-Net.

The implementation follows the manuscript design: an enhanced residual
feature block (ERFB), graph-inspired visual context aggregation (VGCA), and
depth-aware channel gating are combined with a residual projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class ERFB(nn.Module):
    """Enhanced residual feature block used by RFAM."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.skip = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.conv1(x))
        return self.activation(self.conv2(out) + self.skip(x))


class VGCABlock(nn.Module):
    """Lightweight visual graph-context aggregation block."""

    def __init__(self, channels: int):
        super().__init__()
        self.context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.adjacency = nn.Conv2d(channels, channels, 1, bias=True)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.adjacency(x))
        context = self.project(self.context(x) * weights)
        return x + self.norm(context)


class DepthAwareGating(nn.Module):
    """Global channel gate for depth-aware residual feature selection."""

    def __init__(self, channels: int, reduction_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * reduction_ratio))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class RFAM(nn.Module):
    """Hierarchical Residual Feature Alignment Module.

    Args:
        c1: Input channel count.
        c2: Output channel count. Defaults to ``c1``.
        reduction_ratio: Width of the channel-gating bottleneck.
        enhanced_mode: Whether to include VGCA and the alignment FFN. If
            omitted, the larger channel configurations enable this path.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        reduction_ratio: float = 0.25,
        enhanced_mode: bool = False,
    ):
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"c1 must be positive, got {c1}")
        c2 = c1 if c2 is None else int(c2)
        if c2 <= 0:
            raise ValueError(f"c2 must be positive, got {c2}")
        if not 0 < reduction_ratio <= 1:
            raise ValueError(f"reduction_ratio must be in (0, 1], got {reduction_ratio}")

        self.c1, self.c2 = int(c1), c2
        # The supplied experiment YAML did not enable the optional VGCA/FFN
        # branch.  Keep that choice explicit and stable after width scaling.
        self.enhanced_mode = bool(enhanced_mode)
        self.input_projection = Conv(self.c1, self.c2, 1, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.c2, self.c2 * 2, 1, bias=False),
            nn.BatchNorm2d(self.c2 * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c2 * 2, self.c2, 1, bias=False),
            nn.BatchNorm2d(self.c2),
        )
        self.erfb = ERFB(self.c2, self.c2)
        self.gate = DepthAwareGating(self.c2, reduction_ratio)
        self.vgca = VGCABlock(self.c2) if self.enhanced_mode else nn.Identity()
        self.alignment_ffn = (
            nn.Sequential(
                nn.Conv2d(self.c2, self.c2 * 2, 1, bias=False),
                nn.BatchNorm2d(self.c2 * 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.c2 * 2, self.c2, 1, bias=False),
                nn.BatchNorm2d(self.c2),
            )
            if self.enhanced_mode
            else nn.Identity()
        )
        self.skip = nn.Identity() if self.c1 == self.c2 else Conv(self.c1, self.c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        features = self.input_projection(x)
        features = features + self.mlp(features)
        features = self.erfb(features)
        features = self.gate(features)
        features = self.vgca(features)
        features = features + self.alignment_ffn(features)
        return features + residual


__all__ = ("ERFB", "VGCABlock", "DepthAwareGating", "RFAM")
