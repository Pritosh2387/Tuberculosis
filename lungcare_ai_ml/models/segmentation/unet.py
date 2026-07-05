"""
Standard U-Net for lung region segmentation in LungCare AI.

Follows the original Ronneberger et al. (2015) architecture with:
- Four encoder levels (DoubleConv + MaxPool)
- Bottleneck (DoubleConv + MaxPool + Dropout2d)
- Four decoder levels (Upsample/ConvTranspose + skip concat + DoubleConv)
- 1×1 output convolution

Configurable
------------
- ``features``: channel counts at each encoder level.
- ``bilinear``: use bilinear upsampling (no learned transposed conv).
- ``dropout_rate``: spatial dropout applied at the bottleneck.
- ``in_channels`` / ``out_channels``: support greyscale CT and RGB CXR.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("lungcare.models.unet")


# ─── Building blocks ──────────────────────────────────────────────────────────


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv (encoder step)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upsample then concatenate skip connection, then DoubleConv.

    Args:
        in_channels: Total channels **after** concatenation of upsampled
            tensor and skip connection (= upsampled_ch + skip_ch).
        out_channels: Output channels after DoubleConv.
        bilinear: Use bilinear upsample instead of ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Upsampled tensor from the decoder path.
            x2: Skip-connection tensor from the encoder at the same spatial level.
        """
        x1 = self.up(x1)
        # Pad x1 to match x2 if spatial sizes differ (odd-dimension inputs)
        dh = x2.shape[2] - x1.shape[2]
        dw = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ─── Main model ───────────────────────────────────────────────────────────────


class UNet(nn.Module):
    """
    Encoder-decoder U-Net for binary / multi-class segmentation.

    Args:
        in_channels: Input image channels (1 for greyscale, 3 for RGB).
        out_channels: Number of segmentation classes.
        features: Channel counts at encoder levels 1–4.
        bilinear: Use bilinear upsampling (saves parameters vs transposed conv).
        dropout_rate: Spatial dropout probability at bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        bilinear: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        f = list(features)

        # Encoder
        self.inc = DoubleConv(in_channels, f[0])
        self.downs = nn.ModuleList([Down(f[i], f[i + 1]) for i in range(len(f) - 1)])
        self.bottleneck = Down(f[-1], f[-1] * 2 // factor)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Decoder — track running channel count
        ch = f[-1] * 2 // factor
        ups: list[Up] = []
        for i, skip_ch in enumerate(reversed(f)):
            is_last = i == len(f) - 1
            out_ch = skip_ch if is_last else skip_ch // factor
            ups.append(Up(ch + skip_ch, out_ch, bilinear))
            ch = out_ch

        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv2d(ch, out_channels, kernel_size=1)

        logger.info(
            "UNet | in=%d | out=%d | features=%s | bilinear=%s",
            in_channels, out_channels, features, bilinear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor ``(B, in_channels, H, W)``.

        Returns:
            Logit map ``(B, out_channels, H, W)`` — apply sigmoid/softmax
            externally during loss computation.
        """
        # Encoder
        skip_feats: list[torch.Tensor] = []
        x = self.inc(x)
        for down in self.downs:
            skip_feats.append(x)
            x = down(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        # Decoder
        for up, skip in zip(self.ups, reversed(skip_feats)):
            x = up(x, skip)

        return self.outc(x)

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
