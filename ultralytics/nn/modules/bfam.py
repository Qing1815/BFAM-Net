"""Final BFAM-Net modules.

The source experiment directory contained several ablation implementations
under the internal names BFEM, DFRM, and CCGM.  This file exposes the final
paper names and keeps the implementation self-contained so that the public
model configuration does not depend on an ablation-only module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class _SelectiveSSM(nn.Module):
    """Small numerical-stable selective state-space layer for 1-D features."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=False,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1).float()))
        self.D = nn.Parameter(torch.full((self.d_inner,), 0.1))
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x, residual = self.in_proj(x).chunk(2, dim=-1)

        # Explicit asymmetric padding keeps sequence length stable for even kernels.
        left = self.conv1d.kernel_size[0] // 2
        right = self.conv1d.kernel_size[0] - 1 - left
        x = F.pad(x.transpose(1, 2), (left, right))
        x = self.conv1d(x).transpose(1, 2)
        x = F.silu(self.norm(x))

        b_term, c_term = self.x_proj(x).chunk(2, dim=-1)
        delta = torch.clamp(F.softplus(self.dt_proj(x)), min=1e-3, max=0.1)
        a_term = -torch.exp(self.A_log.float()).to(dtype=x.dtype)
        a_bar = torch.exp(torch.clamp(delta.unsqueeze(-1) * a_term.unsqueeze(0).unsqueeze(0), min=-10.0, max=0.0))
        b_bar = torch.clamp(delta.unsqueeze(-1) * b_term.unsqueeze(2), min=-10.0, max=10.0)

        state = x.new_zeros(batch, self.d_inner, self.d_state)
        outputs = []
        for index in range(length):
            state = a_bar[:, index] * state + b_bar[:, index] * x[:, index].unsqueeze(-1)
            state = torch.nan_to_num(state.clamp(-1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
            outputs.append((state * c_term[:, index].unsqueeze(1)).sum(dim=-1))

        y = torch.stack(outputs, dim=1)
        y = y + x * self.D.view(1, 1, -1)
        y = y * F.silu(residual)
        return self.out_proj(torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3))


class _AxisSSM(nn.Module):
    """Bidirectional SSM scanning along one image axis."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels * 2, 8)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden)
        self.forward_ssm = _SelectiveSSM(hidden, expand=1)
        self.backward_ssm = _SelectiveSSM(hidden, expand=1)
        self.output_norm = nn.LayerNorm(hidden)
        self.project = nn.Linear(hidden, channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = self.expand(self.norm(sequence))
        forward = self.forward_ssm(sequence)
        backward = torch.flip(self.backward_ssm(torch.flip(sequence, dims=[1])), dims=[1])
        return self.project(self.output_norm((forward + backward) * 0.5))


class _BidirectionalSpatialSSM(nn.Module):
    """Apply independent horizontal and vertical bidirectional scans."""

    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = _AxisSSM(channels)
        self.vertical = _AxisSSM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        horizontal = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        horizontal = self.horizontal(horizontal).reshape(batch, height, width, channels).permute(0, 3, 1, 2)

        vertical = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        vertical = self.vertical(vertical).reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        return horizontal + vertical


class BFRE(nn.Module):
    """Bidirectional Feature Representation Enhancement module.

    Directional SSM scans are performed after one-dimensional global pooling.
    Their interaction is implemented as a linear spatial gate, which avoids a
    quadratic ``HW x HW`` attention tensor at the high-resolution P3 level.
    """

    def __init__(self, c1: int, c2: int | None = None, e: float = 0.5):
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(int(c1 * e), 8)
        self.project_in = Conv(c1, hidden, 1, 1)
        self.horizontal = _AxisSSM(hidden)
        self.vertical = _AxisSSM(hidden)
        self.project_out = Conv(hidden, c2, 1, 1)
        self.skip = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        features = self.project_in(x)
        _, _, height, width = features.shape
        horizontal = features.mean(dim=2, keepdim=True).expand(-1, -1, height, -1)
        vertical = features.mean(dim=3, keepdim=True).expand(-1, -1, -1, width)

        horizontal = self.horizontal(horizontal.permute(0, 2, 3, 1).reshape(-1, width, features.shape[1]))
        horizontal = horizontal.reshape(features.shape[0], height, width, features.shape[1]).permute(0, 3, 1, 2)
        vertical = self.vertical(vertical.permute(0, 3, 2, 1).reshape(-1, height, features.shape[1]))
        vertical = vertical.reshape(features.shape[0], width, height, features.shape[1]).permute(0, 3, 2, 1)

        gate = torch.sigmoid(horizontal + vertical)
        return self.project_out(features * gate) + residual


class _ChannelContext(nn.Module):
    """Channel-context branch used by BCFM.

    The attention matrix is channel-by-channel (``C x C``), matching the
    reference DC-channel attention.  Keeping the spatial dimension in the
    feature/value tensors avoids constructing a quadratic ``HW x HW`` matrix.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.key = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query = self.query(x).flatten(2)
        key = self.key(x).flatten(2)
        value = self.value(x).flatten(2)
        # Scale by the dot-product width, as in channel attention.  This is
        # deliberately independent of the number of spatial locations.
        attention = torch.softmax(torch.matmul(query, key.transpose(1, 2)) / math.sqrt(height * width), dim=-1)
        return self.output(torch.matmul(attention, value).reshape(batch, channels, height, width))


class BCFM(nn.Module):
    """Bidirectional Context Feature Optimization module."""

    def __init__(self, c1: int, c2: int | None = None, e: float = 0.5):
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(int(c1 * e), 8)
        self.input = Conv(c1, hidden, 1, 1)
        self.channel_context = _ChannelContext(hidden)
        self.spatial_context = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            _BidirectionalSpatialSSM(hidden),
        )
        self.output = Conv(hidden, c2, 1, 1, act=False)
        self.skip = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        features = self.input(x)
        refined = self.channel_context(features) + self.spatial_context(features)
        return self.output(refined) + residual


class _DynamicUpsample(nn.Module):
    """Grouped learned-offset sampling used by CSCG.

    The offset layout follows the supplied DySample reference: every channel
    group receives an offset for each sub-pixel location, giving
    ``2 * groups * scale^2`` predicted offsets per input pixel.
    """

    def __init__(self, channels: int, scale_factor: int = 2, groups: int = 4):
        super().__init__()
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
        self.scale_factor = scale_factor
        self.groups = groups
        self.offset = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2 * groups * scale_factor * scale_factor, 1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        out_height, out_width = height * self.scale_factor, width * self.scale_factor
        offset = self.offset(x)
        offset = offset.view(
            batch,
            self.groups,
            2,
            self.scale_factor,
            self.scale_factor,
            height,
            width,
        )
        offset = offset.permute(0, 1, 5, 3, 6, 4, 2).contiguous()
        offset = offset.view(batch, self.groups, out_height, out_width, 2)

        # Build the base grid in input-pixel coordinates, then normalize it
        # for grid_sample with align_corners=True.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(out_height, device=x.device, dtype=torch.float32) / self.scale_factor,
            torch.arange(out_width, device=x.device, dtype=torch.float32) / self.scale_factor,
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).unsqueeze(0)
        grid = grid + offset.float()
        if width > 1:
            grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
        else:
            grid[..., 0] = 0
        if height > 1:
            grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
        else:
            grid[..., 1] = 0

        grouped_x = x.reshape(batch * self.groups, channels // self.groups, height, width)
        grouped_grid = grid.expand(batch, self.groups, -1, -1, -1).reshape(
            batch * self.groups, out_height, out_width, 2
        )
        sampled = F.grid_sample(
            grouped_x.float(),
            grouped_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.to(dtype=x.dtype).reshape(batch, channels, out_height, out_width)


class CSCG(nn.Module):
    """Contextual Semantic Content Guidance module for feature fusion."""

    accepts_multiple_inputs = True

    def __init__(self, low_channels: int, high_channels: int, out_channels: int | None = None, scale_factor: int = 2):
        super().__init__()
        out_channels = high_channels if out_channels is None else out_channels
        self.low_preprocess = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 3, padding=1, groups=low_channels, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.SiLU(inplace=True),
        )
        self.upsample = _DynamicUpsample(low_channels, scale_factor)
        self.high_preprocess = nn.Sequential(
            nn.Conv2d(high_channels, high_channels, 3, padding=1, groups=high_channels, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.SiLU(inplace=True),
        )
        self.high_mlp = nn.Sequential(
            nn.Conv2d(high_channels, high_channels * 2, 1, bias=False),
            nn.BatchNorm2d(high_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(high_channels * 2, high_channels, 1, bias=False),
            nn.BatchNorm2d(high_channels),
        )
        self.fusion = Conv(low_channels + high_channels, out_channels, 1, 1)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, low_res_feat: torch.Tensor, high_res_feat: torch.Tensor) -> torch.Tensor:
        low = self.upsample(self.low_preprocess(low_res_feat))
        high = self.high_preprocess(high_res_feat)
        high = high + self.high_mlp(high)
        if low.shape[2:] != high.shape[2:]:
            low = F.interpolate(low, size=high.shape[2:], mode="bilinear", align_corners=False)

        output = self.output(self.fusion(torch.cat((low, high), dim=1)))
        if output.shape[1] == high_res_feat.shape[1]:
            output = output + high_res_feat
        return output


__all__ = ("BFRE", "BCFM", "CSCG")
