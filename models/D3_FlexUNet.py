#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SwinUNETR 3D (UNet3D-style wrapper)
# - accepts input as [B, C, H, W, D] (like your UNet3D demo) OR [B, C, D, H, W]
# - internally converts to [B, C, D, H, W]
# - pads D/H/W to multiples of 32 (required by SwinUNETR patch merging)
# - outputs logits as [B, out_channels, H, W, D] (match your pipeline habit)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import FlexUNet


def _pad_to_multiple_3d(x: torch.Tensor, multiple: int = 32):
    """
    Pad tensor x on (D,H,W) so each is divisible by `multiple`.
    x: [B, C, D, H, W]
    return:
      x_pad, (D0,H0,W0), (Dp,Hp,Wp), pad_tuple_for_F_pad
    """
    assert x.ndim == 5, f"expect 5D [B,C,D,H,W], got {x.shape}"
    B, C, D, H, W = x.shape

    def up_to(v):
        return int(math.ceil(v / multiple) * multiple)

    Dp, Hp, Wp = up_to(D), up_to(H), up_to(W)
    pd, ph, pw = Dp - D, Hp - H, Wp - W

    # F.pad order for 5D is (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = (pw // 2, pw - pw // 2,
           ph // 2, ph - ph // 2,
           pd // 2, pd - pd // 2)

    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(x, pad, mode="constant", value=0.0)

    return x, (D, H, W), (Dp, Hp, Wp), pad


def _unpad_to_original_3d(x: torch.Tensor, original_dhw, pad):
    """
    Crop back to original size (D,H,W) using the same symmetric pad used in _pad_to_multiple_3d.
    x: [B, C, Dp, Hp, Wp]
    """
    D, H, W = original_dhw
    pw_l, pw_r, ph_l, ph_r, pd_l, pd_r = pad

    # After padding, spatial dims are:
    # Dp = D + pd_l + pd_r, etc.
    d0, d1 = pd_l, pd_l + D
    h0, h1 = ph_l, ph_l + H
    w0, w1 = pw_l, pw_l + W
    return x[:, :, d0:d1, h0:h1, w0:w1]


class SwinUNETR3D(nn.Module):
    """
    Engineering-stable SwinUNETR wrapper for 3D medical image segmentation.

    Accepted input formats:
      - [B, C, H, W, D]  (your UNet3D demo style)
      - [B, C, D, H, W]  (MONAI/SwinUNETR native)
    Output:
      - logits [B, out_channels, H, W, D]  (match your UNet3D demo habit)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        img_size=(128, 128, 128),   # only used by SwinUNETR init; runtime will auto-pad anyway
        feature_size: int = 12,     # must be divisible by 12
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        patch_size: int = 2,
        pad_multiple: int = 32,     # SwinUNETR requirement for patch merging (2**5 when patch_size=2)
    ):
        super().__init__()
        self.pad_multiple = pad_multiple

        self.net = FlexUNet(
            # img_size=128,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=3, pretrained=False,backbone='resnet10'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept [B,C,H,W,D] or [B,C,D,H,W]
        if x.ndim != 5:
            raise ValueError(f"Expect 5D input, got shape: {tuple(x.shape)}")

        # Heuristic: if last dim is D and middle dims are H/W (common in your code)
        # x: [B,C,H,W,D] -> [B,C,D,H,W]
        # If already [B,C,D,H,W], keep it.
        # We decide based on which axis is likely depth: usually D is smaller than H/W,
        # but to be safe, we just follow your previous convention:
        # if x is [B,C,H,W,D], the last dim is D.
        # We'll treat input as [B,C,H,W,D] by default and convert.
        # If user already provided [B,C,D,H,W], they can pass `x = x.contiguous()` with that order
        # and set a flag; but we keep it simple: detect by comparing axis sizes.
        B, C, a, b, c = x.shape
        # If (a,b) look like H,W and c looks like D (often smaller), convert.
        # Otherwise assume it's already [B,C,D,H,W].
        # if c <= a and c <= b:
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B,C,D,H,W]

        # pad to multiple-of-32 for stable forward
        # x_pad, orig_dhw, _, pad = _pad_to_multiple_3d(x, multiple=self.pad_multiple)

        y = self.net(x)  # [B,out,Dp,Hp,Wp]

        # unpad back to original D/H/W
        # y = _unpad_to_original_3d(y, orig_dhw, pad)  # [B,out,D,H,W]

        # return in [B,out,H,W,D] to match your UNet3D demo style
        y = y.permute(0, 1, 3, 4, 2).contiguous()
        return y


if __name__ == "__main__":
    # like your UNet3D demo: create [B,C,H,W,D], then model handles permutation internally
    B, C, H, W, D = 2, 1, 128, 128, 128
    x = torch.randn(B, C, H, W, D).cuda()

    model = SwinUNETR3D(
        in_channels=1,
        out_channels=4,
        img_size=(128, 128, 128),
        feature_size=48,     # must be divisible by 12
        patch_size=2,
        pad_multiple=32,
        use_checkpoint=False
    ).cuda()

    y = model(x)
    print("Output shape:", y.shape)  # [B,4,H,W,D] == [2,4,128,128,128]

    # THOP profiling (may not fully support some ops; if it errors, that's normal)
    try:
        from thop import profile
        flops, params = profile(model, (x,), verbose=False)
        print("FLOPs(G):", flops / 1e9)
        print("Params(M):", params / 1e6)
    except Exception as e:
        print("[WARN] THOP failed:", type(e).__name__, e)
