#!/usr/bin/env python
# -*- coding: utf-8 -*-
# UNext3D (UNet3D-style)
# - 3D CNN encoder/decoder (Conv3d + InstanceNorm3d)
# - Downsample: stride=2 on (D,H,W) like your UNet3D
# - Upsample: nearest + 1x1 reduce + concat skip like your UNet3D
# - Bottleneck: 3D shifted-MLP blocks (UNext idea extended to 3D)
# Input : [B, C, D, H, W]
# Output: [B, n_classes, D, H, W] (logits)

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------
# utils
# ---------------------------
def match_size_3d(x, ref):
    """
    Pad + center crop to make x match ref on (D,H,W).
    x/ref: [B,C,D,H,W]
    """
    _, _, d, h, w = x.shape
    _, _, rd, rh, rw = ref.shape

    pd = max(rd - d, 0)
    ph = max(rh - h, 0)
    pw = max(rw - w, 0)
    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(
            x,
            (pw // 2, pw - pw // 2,
             ph // 2, ph - ph // 2,
             pd // 2, pd - pd // 2)
        )

    _, _, d, h, w = x.shape
    sd = max((d - rd) // 2, 0)
    sh = max((h - rh) // 2, 0)
    sw = max((w - rw) // 2, 0)
    return x[:, :, sd:sd + rd, sh:sh + rh, sw:sw + rw]


# ---------------------------
# UNet3D-style blocks
# ---------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.Dropout3d(p) if p > 0 else nn.Identity(),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    """Downsample on (D,H,W) by 2, like your UNet3D."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up3D(nn.Module):
    """Upsample by 2, reduce channels by 1x1, then concat skip."""
    def __init__(self, ch):
        super().__init__()
        self.reduce = nn.Conv3d(ch, ch // 2, 1, 1, bias=False)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.reduce(x)
        x = match_size_3d(x, skip)
        return torch.cat([x, skip], dim=1)


# ---------------------------
# UNext core: shifted MLP (3D)
# ---------------------------
class DWConv3D(nn.Module):
    """Depthwise conv3d used inside shift-MLP."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

    def forward(self, x):
        return self.dwconv(x)


def shift_along_dim(x, shift, dim):
    """x: [B,C,D,H,W], shift channels by rolling along given spatial dim."""
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=dim)


class ShiftMLP3D(nn.Module):
    """
    3D shifted-MLP:
    - channel split into shift_size groups
    - shift along D, then H, then W
    - DWConv3d for local mixing
    - pointwise MLP (1x1 linear on channel via conv1x1 equivalent)
    """
    def __init__(self, dim, mlp_ratio=4.0, shift_size=5, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.fc1 = nn.Conv3d(dim, hidden_dim, kernel_size=1, bias=True)
        self.dwconv = DWConv3D(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv3d(hidden_dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B,C,D,H,W]
        B, C, D, H, W = x.shape

        # pad for safe shift
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad))  # pad W,H,D
        _, _, Dp, Hp, Wp = x.shape

        # 1) shift along D
        chunks = torch.chunk(x, self.shift_size, dim=1) if C % self.shift_size == 0 else torch.chunk(x, self.shift_size, dim=1)
        shifted = []
        for i, c in enumerate(chunks):
            s = i - self.shift_size // 2
            shifted.append(shift_along_dim(c, s, dim=2))  # dim=2 is D
        x = torch.cat(shifted, dim=1)

        # 2) shift along H
        chunks = torch.chunk(x, self.shift_size, dim=1)
        shifted = []
        for i, c in enumerate(chunks):
            s = i - self.shift_size // 2
            shifted.append(shift_along_dim(c, s, dim=3))  # dim=3 is H
        x = torch.cat(shifted, dim=1)

        # 3) shift along W
        chunks = torch.chunk(x, self.shift_size, dim=1)
        shifted = []
        for i, c in enumerate(chunks):
            s = i - self.shift_size // 2
            shifted.append(shift_along_dim(c, s, dim=4))  # dim=4 is W
        x = torch.cat(shifted, dim=1)

        # crop back to original D/H/W
        x = x[:, :, self.pad:self.pad + D, self.pad:self.pad + H, self.pad:self.pad + W]

        # MLP + DWConv
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ShiftedBlock3D(nn.Module):
    """LayerNorm3d(approx) + ShiftMLP3D + residual."""
    def __init__(self, dim, mlp_ratio=4.0, shift_size=5, drop=0.0):
        super().__init__()
        # 用 InstanceNorm3d 更贴近你 UNet3D 的稳定风格（小 batch 更稳）
        self.norm = nn.InstanceNorm3d(dim, affine=True)
        self.mlp = ShiftMLP3D(dim=dim, mlp_ratio=mlp_ratio, shift_size=shift_size, drop=drop)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


# ---------------------------
# UNext3D (UNet3D-style)
# ---------------------------
class UNext3D(nn.Module):
    """
    UNet3D-style encoder-decoder + UNext shifted-MLP bottleneck (3D).
    Input : [B, inchannel, D, H, W]
    Output: [B, n_classes, D, H, W] (logits)
    """
    def __init__(self, inchannel=1, n_classes=4, base_ch=8, dropout=0.0,
                 mlp_ratio=4.0, shift_size=5, bottleneck_depth=2):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        # Encoder (same as UNet3D style)
        self.enc1 = ConvBlock3D(inchannel, c1, p=dropout)
        self.down1 = Down3D(c1)

        self.enc2 = ConvBlock3D(c1, c2, p=dropout)
        self.down2 = Down3D(c2)

        self.enc3 = ConvBlock3D(c2, c3, p=dropout)
        self.down3 = Down3D(c3)

        self.enc4 = ConvBlock3D(c3, c4, p=dropout)
        self.down4 = Down3D(c4)

        # Bottleneck conv
        self.bottleneck = ConvBlock3D(c4, c5, p=dropout)

        # UNext core at bottleneck: shifted-MLP blocks (3D)
        self.mlp_blocks = nn.Sequential(*[
            ShiftedBlock3D(dim=c5, mlp_ratio=mlp_ratio, shift_size=shift_size, drop=dropout)
            for _ in range(bottleneck_depth)
        ])

        # Decoder (UNet3D style: up + concat + conv)
        self.up1 = Up3D(c5)                  # reduce to c4 then concat with c4 => c5
        self.dec1 = ConvBlock3D(c5, c4, p=dropout)

        self.up2 = Up3D(c4)                  # reduce to c3 then concat with c3 => c4
        self.dec2 = ConvBlock3D(c4, c3, p=dropout)

        self.up3 = Up3D(c3)                  # reduce to c2 then concat with c2 => c3
        self.dec3 = ConvBlock3D(c3, c2, p=dropout)

        self.up4 = Up3D(c2)                  # reduce to c1 then concat with c1 => c2
        self.dec4 = ConvBlock3D(c2, c1, p=dropout)

        self.out = nn.Conv3d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        r1 = self.enc1(x)
        r2 = self.enc2(self.down1(r1))
        r3 = self.enc3(self.down2(r2))
        r4 = self.enc4(self.down3(r3))
        r5 = self.bottleneck(self.down4(r4))

        # UNext shifted-MLP bottleneck enhancement
        r5 = self.mlp_blocks(r5)

        # Decoder
        o1 = self.dec1(self.up1(r5, r4))
        o2 = self.dec2(self.up2(o1, r3))
        o3 = self.dec3(self.up3(o2, r2))
        o4 = self.dec4(self.up4(o3, r1))

        return self.out(o4)  # logits [B, n_classes, D, H, W]


# ---------------------------
# example like UNet3D
# ---------------------------
if __name__ == "__main__":
    # 假设原始 patch = (H,W,D) = (128,128,128)，并且你的数据最初是 [B,C,H,W,D]
    B, C, H, W, D = 2, 1, 128, 128, 128

    x = torch.randn(B, C, H, W, D).cuda()
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # -> [B,1,D,H,W]

    model = UNext3D(inchannel=1, n_classes=4, base_ch=8, dropout=0.1,
                    mlp_ratio=4.0, shift_size=5, bottleneck_depth=2).cuda()

    y = model(x)
    print("Output shape:", y.shape)  # [B,4,D,H,W]

    # THOP profiling
    from thop import profile
    flops, params = profile(model, (x,), verbose=False)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
