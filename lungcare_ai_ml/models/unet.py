"""
models/unet.py
───────────────
Standard U-Net for binary lung segmentation.

Architecture (for 224×224 input, 1 output channel):
    Encoder: 4 × (Conv2d → BN → ReLU) × 2, MaxPool2d
    Bottleneck: double conv at lowest resolution
    Decoder: 4 × (Upsample + skip concat + double conv)
    Head: Conv2d(64, out_channels=1, kernel_size=1)  ← binary mask logit

Skip connections preserve spatial detail lost during downsampling —
critical for precise lung boundary segmentation.

Returns raw logits. Apply sigmoid + threshold at inference time.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("lungcare.models.unet")


class _DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BatchNorm → ReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """MaxPool2d → DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """Bilinear upsample + skip-connection concatenation + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims don't match (odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Standard U-Net for binary lung segmentation.

    Args:
        in_channels:  Input image channels (1 = greyscale, 3 = RGB).
        out_channels: Output mask channels (1 = binary segmentation).
        base_filters:  Channel count in the first encoder block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _Down(f,      f * 2)
        self.enc3 = _Down(f * 2,  f * 4)
        self.enc4 = _Down(f * 4,  f * 8)
        # Bottleneck
        self.bottleneck = _Down(f * 8, f * 16)
        # Decoder
        self.dec4 = _Up(f * 16 + f * 8, f * 8)
        self.dec3 = _Up(f * 8  + f * 4, f * 4)
        self.dec2 = _Up(f * 4  + f * 2, f * 2)
        self.dec1 = _Up(f * 2  + f,     f)
        # Output head
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

        logger.info("UNet | in=%d | out=%d | base_filters=%d",
                    in_channels, out_channels, base_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (B, in_channels, H, W).

        Returns:
            Logit map (B, out_channels, H, W). Apply sigmoid for probabilities.
        """
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x  = self.bottleneck(s4)
        x  = self.dec4(x, s4)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)
        return self.head(x)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
