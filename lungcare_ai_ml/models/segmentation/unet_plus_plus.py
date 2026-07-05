"""
U-Net++ (Nested U-Net) for multi-disease segmentation in LungCare AI.

Implements dense nested skip connections between all encoder/decoder nodes
at the same spatial resolution, plus optional deep supervision across the
four intermediate output maps.

Reference
---------
Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
Segmentation", DLMIA 2018.  https://arxiv.org/abs/1807.10165
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation.unet import DoubleConv

logger = logging.getLogger("lungcare.models.unet_plus_plus")


class _UpsampleConv(nn.Module):
    """Bilinear upsample followed by a 1×1 channel reduction conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ with dense skip connections and optional deep supervision.

    Architecture
    ------------
    For ``depth`` encoder levels and feature counts ``F = [f₀, f₁, …, f_d]``:

    - Encoder nodes X[i][0] for i = 0..d
    - Dense decoder nodes X[i][j] for j = 1..d-i, where each node
      receives all previous same-level outputs X[i][0..j-1] and the
      upsampled output from one level below X[i+1][j-1].

    Deep supervision
    ----------------
    During **training**, ``forward`` returns a list of logit maps, one per
    decoder output node at level 0: ``[X[0][1], X[0][2], …, X[0][depth]]``.
    During **inference** (``model.eval()``), only the final output
    ``X[0][depth]`` is returned.

    Args:
        in_channels: Input image channels.
        out_channels: Segmentation classes.
        features: Feature counts per encoder level.  The bottleneck
            (deepest) level uses ``features[-1]``.
        deep_supervision: Enable multi-output training mode.
        dropout_rate: Spatial dropout at the deepest encoder node.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256, 512),
        deep_supervision: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        f = list(features)
        depth = len(f)

        # ── Encoder X[i][0] ───────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.encoders.append(DoubleConv(in_channels, f[0]))
        for i in range(1, depth):
            self.encoders.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(f[i - 1], f[i]),
                )
            )

        self.dropout = nn.Dropout2d(p=dropout_rate)

        # ── Dense decoder nodes X[i][j] for j >= 1 ────────────────────────────
        # X[i][j] input channels: (j * f[i]) + f[i+1]
        # (j previous same-level + 1 upsampled from below)
        # Output channels: f[i]
        self.dense_convs: nn.ModuleDict = nn.ModuleDict()
        self.upsamples: nn.ModuleDict = nn.ModuleDict()

        for j in range(1, depth):
            for i in range(depth - j):
                in_ch = j * f[i] + f[i + 1]
                self.dense_convs[f"{i}_{j}"] = DoubleConv(in_ch, f[i])
                self.upsamples[f"{i}_{j}"] = _UpsampleConv(f[i + 1], f[i + 1])

        # ── Output heads ──────────────────────────────────────────────────────
        # One per decoder output at level i=0 (X[0][1..depth-1])
        if deep_supervision:
            self.output_convs = nn.ModuleList([
                nn.Conv2d(f[0], out_channels, kernel_size=1)
                for _ in range(1, depth)
            ])
        else:
            self.output_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)

        logger.info(
            "UNetPlusPlus | in=%d | out=%d | features=%s | deep_supervision=%s",
            in_channels, out_channels, features, deep_supervision,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            x: Input tensor ``(B, in_channels, H, W)``.

        Returns:
            - During **training** with ``deep_supervision=True``:
              list of ``depth-1`` logit maps, coarsest to finest.
            - Otherwise: single logit map ``(B, out_channels, H, W)``.
        """
        depth = len(self.encoders)

        # Build nodes dict: nodes[i][j] = feature tensor for X[i][j]
        nodes: list[list[torch.Tensor | None]] = [
            [None] * depth for _ in range(depth)
        ]

        # Encoder pass
        enc_in = x
        for i, encoder in enumerate(self.encoders):
            enc_in = encoder(enc_in)
            if i == depth - 1:
                enc_in = self.dropout(enc_in)
            nodes[i][0] = enc_in

        # Dense decoder pass
        for j in range(1, depth):
            for i in range(depth - j):
                key = f"{i}_{j}"
                # Upsample from node below: X[i+1][j-1]
                up = self.upsamples[key](nodes[i + 1][j - 1])   # type: ignore[index]
                # Collect all previous same-level nodes: X[i][0..j-1]
                same_level = [nodes[i][k] for k in range(j)]    # type: ignore[misc]
                # Pad upsampled to match spatial size of same_level[0]
                target_h, target_w = same_level[0].shape[2:]    # type: ignore[union-attr]
                if up.shape[2:] != (target_h, target_w):
                    dh = target_h - up.shape[2]
                    dw = target_w - up.shape[3]
                    up = F.pad(up, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
                cat = torch.cat(same_level + [up], dim=1)        # type: ignore[list-item]
                nodes[i][j] = self.dense_convs[key](cat)

        # Output
        if self.training and self.deep_supervision:
            return [
                self.output_convs[j](nodes[0][j + 1])            # type: ignore[index]
                for j in range(depth - 1)
            ]
        final = nodes[0][depth - 1]                               # type: ignore[index]
        if self.deep_supervision:
            return self.output_convs[-1](final)
        return self.output_conv(final)

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
