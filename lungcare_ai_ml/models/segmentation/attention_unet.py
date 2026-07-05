"""
Attention U-Net for lung lesion segmentation in LungCare AI.

Extends standard U-Net with Attention Gates (Oktay et al., 2018) at
every decoder skip connection.  Each gate learns to suppress irrelevant
encoder features before they are concatenated into the decoder path,
improving focus on small lesions (nodules, infiltrates, cavities).

Reference
---------
Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
MIDL 2018.  https://arxiv.org/abs/1804.03999
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation.unet import DoubleConv, Down

logger = logging.getLogger("lungcare.models.attention_unet")


# ─── Attention Gate ───────────────────────────────────────────────────────────


class AttentionGate(nn.Module):
    """
    Soft Attention Gate that filters skip-connection features.

    Computes a spatial attention map α ∈ [0, 1] conditioned on both
    the gating signal *g* (from the decoder) and the skip features *x*
    (from the encoder).

    Args:
        F_g: Channel count of the gating signal *g* (decoder features).
        F_l: Channel count of the skip features *x* (encoder features).
        F_int: Intermediate channel count for the additive attention computation.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal ``(B, F_g, H, W)`` — from decoder path (smaller).
            x: Skip features ``(B, F_l, H', W')`` — from encoder (larger).

        Returns:
            Gated skip features ``(B, F_l, H', W')``.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample gating signal to match skip feature spatial size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


# ─── Decoder block with attention ─────────────────────────────────────────────


class AttentionUp(nn.Module):
    """
    Decoder step with an Attention Gate on the skip connection.

    Args:
        in_channels: Total channels after concatenation (skip_gated + up).
        out_channels: Output channels.
        F_int: Intermediate channels for the Attention Gate.
        bilinear: Use bilinear upsample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        F_int: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        skip_ch = in_channels // 2
        up_ch = in_channels // 2
        self.attention = AttentionGate(F_g=up_ch, F_l=skip_ch, F_int=F_int)
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.up = nn.ConvTranspose2d(up_ch, up_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Decoder tensor (gating signal).
            x2: Encoder skip tensor.
        """
        x1 = self.up(x1)
        x2 = self.attention(g=x1, x=x2)
        dh = x2.shape[2] - x1.shape[2]
        dw = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ─── Main model ───────────────────────────────────────────────────────────────


class AttentionUNet(nn.Module):
    """
    Attention U-Net encoder-decoder with soft Attention Gates.

    Identical to :class:`UNet` in structure but replaces each vanilla Up
    block with :class:`AttentionUp` which applies an Attention Gate to the
    skip connection before concatenation.

    Args:
        in_channels: Input channels (1 = greyscale, 3 = RGB).
        out_channels: Segmentation classes.
        features: Encoder channel counts at each level.
        bilinear: Use bilinear upsampling.
        dropout_rate: Spatial dropout at the bottleneck.
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
        factor = 2 if bilinear else 1
        f = list(features)

        # Encoder (same as U-Net)
        self.inc = DoubleConv(in_channels, f[0])
        self.downs = nn.ModuleList([Down(f[i], f[i + 1]) for i in range(len(f) - 1)])
        self.bottleneck = Down(f[-1], f[-1] * 2 // factor)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Decoder with Attention Gates
        ch = f[-1] * 2 // factor
        ups: list[AttentionUp] = []
        for i, skip_ch in enumerate(reversed(f)):
            is_last = i == len(f) - 1
            out_ch = skip_ch if is_last else skip_ch // factor
            F_int = max(skip_ch // 4, 8)
            ups.append(
                AttentionUp(
                    in_channels=ch + skip_ch,
                    out_channels=out_ch,
                    F_int=F_int,
                    bilinear=bilinear,
                )
            )
            ch = out_ch

        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv2d(ch, out_channels, kernel_size=1)

        logger.info(
            "AttentionUNet | in=%d | out=%d | features=%s | bilinear=%s",
            in_channels, out_channels, features, bilinear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor ``(B, in_channels, H, W)``.

        Returns:
            Logit map ``(B, out_channels, H, W)``.
        """
        skip_feats: list[torch.Tensor] = []
        x = self.inc(x)
        for down in self.downs:
            skip_feats.append(x)
            x = down(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        for up, skip in zip(self.ups, reversed(skip_feats)):
            x = up(x, skip)

        return self.outc(x)

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Return the attention weight maps α from each decoder Attention Gate.

        Runs a full forward pass with hooks and returns a list of attention
        tensors ``(B, 1, H_i, W_i)`` ordered from deepest (coarsest) to
        shallowest (finest) decoder level.

        Args:
            x: Input tensor.

        Returns:
            List of attention maps, one per decoder level.
        """
        attention_maps: list[torch.Tensor] = []
        hooks = []
        for up_block in self.ups:
            def make_hook(attn_list: list[torch.Tensor]):
                def hook(m: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                    attn_list.append(out.detach())
                return hook
            hooks.append(up_block.attention.psi.register_forward_hook(make_hook(attention_maps)))

        with torch.no_grad():
            _ = self.forward(x)

        for h in hooks:
            h.remove()

        return attention_maps

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
