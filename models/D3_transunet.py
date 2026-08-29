#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 3D TransUNet (UNet3D-style) with slice-wise ViT at bottleneck
# Input : [B, C, D, H, W]
# Output: [B, n_classes, D, H, W] (logits)

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# IMPORTANT: keep your original ViT unchanged
from model2.D2_ViT import ViT


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
    return x[:, :, sd:sd+rd, sh:sh+rh, sw:sw+rw]


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.1):
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
    """UNet3D-style: downsample D/H/W all by 2"""
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
    """UNet3D-style: upsample by 2 then reduce channels by 1x1 conv, then concat skip"""
    def __init__(self, ch):
        super().__init__()
        self.reduce = nn.Conv3d(ch, ch // 2, 1, 1, bias=False)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.reduce(x)
        x = match_size_3d(x, skip)
        return torch.cat([x, skip], dim=1)


class SliceWiseViT3D(nn.Module):
    """
    Apply 2D ViT slice-wise on the bottleneck feature map.
    Input : [B, C, D, H, W]
    Output: [B, C, D, H, W]
    """
    def __init__(self, vit_img_dim: int, in_channels: int,
                 head_num: int = 4, mlp_dim: int = 256, block_num: int = 4):
        super().__init__()
        self.vit_img_dim = vit_img_dim  # H5 (assume H5==W5)
        self.vit = ViT(
            img_dim=vit_img_dim,
            in_channels=in_channels,
            embedding_dim=in_channels,
            head_num=head_num,
            mlp_dim=mlp_dim,
            block_num=block_num,
            patch_dim=1,
            classification=False
        )

    def forward(self, x):
        # x: [B,C,D,H,W]
        B, C, D, H, W = x.shape

        # ensure square for this ViT implementation (common in such code)
        if H != self.vit_img_dim or W != self.vit_img_dim:
            # align to (vit_img_dim, vit_img_dim) by pad+crop
            ref = torch.empty((B, C, D, self.vit_img_dim, self.vit_img_dim), device=x.device, dtype=x.dtype)
            x = match_size_3d(x, ref)
            B, C, D, H, W = x.shape

        # slice-wise: (B*D, C, H, W)
        x2d = rearrange(x, "b c d h w -> (b d) c h w")
        tokens = self.vit(x2d)  # expected [B*D, (H*W), C]
        x_out = rearrange(tokens, "(b d) (h w) c -> b c d h w",
                          b=B, d=D, h=self.vit_img_dim, w=self.vit_img_dim)
        return x_out


class TransUNet3D(nn.Module):
    """
    3D TransUNet with UNet3D-style encoder/decoder + slice-wise ViT at bottleneck.
    Input : [B, C, D, H, W]
    Output: [B, n_classes, D, H, W] (logits)
    """
    def __init__(
        self,
        inchannel=1,
        n_classes=4,
        base_ch=8,
        dropout=0.1,
        # ViT config
        img_h=128,               # input H (used to compute bottleneck H/16)
        head_num=4,
        mlp_dim=256,
        block_num=4,
    ):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        # Encoder (same spirit as UNet3D)
        self.enc1 = ConvBlock3D(inchannel, c1, p=dropout)
        self.down1 = Down3D(c1)

        self.enc2 = ConvBlock3D(c1, c2, p=dropout)
        self.down2 = Down3D(c2)

        self.enc3 = ConvBlock3D(c2, c3, p=dropout)
        self.down3 = Down3D(c3)

        self.enc4 = ConvBlock3D(c3, c4, p=dropout)
        self.down4 = Down3D(c4)

        self.bottleneck = ConvBlock3D(c4, c5, p=dropout)

        # ViT at bottleneck (slice-wise on H/16 x W/16)
        vit_img_dim = img_h // 16
        self.vit3d = SliceWiseViT3D(
            vit_img_dim=vit_img_dim,
            in_channels=c5,
            head_num=head_num,
            mlp_dim=mlp_dim,
            block_num=block_num
        )

        # Decoder (same structure as UNet3D)
        self.up1 = Up3D(c5)
        self.dec1 = ConvBlock3D(c5, c4, p=dropout)

        self.up2 = Up3D(c4)
        self.dec2 = ConvBlock3D(c4, c3, p=dropout)

        self.up3 = Up3D(c3)
        self.dec3 = ConvBlock3D(c3, c2, p=dropout)

        self.up4 = Up3D(c2)
        self.dec4 = ConvBlock3D(c2, c1, p=dropout)

        self.out = nn.Conv3d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # x: [B,C,D,H,W]
        r1 = self.enc1(x)                 # [B,c1,D,H,W]
        r2 = self.enc2(self.down1(r1))    # [B,c2,D/2,H/2,W/2]
        r3 = self.enc3(self.down2(r2))    # [B,c3,D/4,H/4,W/4]
        r4 = self.enc4(self.down3(r3))    # [B,c4,D/8,H/8,W/8]
        r5 = self.bottleneck(self.down4(r4))  # [B,c5,D/16,H/16,W/16]

        # ViT enhancement at bottleneck
        r5 = self.vit3d(r5)               # [B,c5,D/16,H/16,W/16]

        o1 = self.dec1(self.up1(r5, r4))  # -> [B,c4,D/8,H/8,W/8]
        o2 = self.dec2(self.up2(o1, r3))  # -> [B,c3,D/4,H/4,W/4]
        o3 = self.dec3(self.up3(o2, r2))  # -> [B,c2,D/2,H/2,W/2]
        o4 = self.dec4(self.up4(o3, r1))  # -> [B,c1,D,H,W]

        return self.out(o4)               # logits [B, n_classes, D, H, W]


if __name__ == "__main__":
    # 与 UNet3D 类似的输入样例：
    # 这里假设原始数据是 [B,C,H,W,D]，然后 permute 成 [B,C,D,H,W]
    B, C, H, W, D = 2, 1, 128, 128, 128

    x = torch.randn(B, C, H, W, D).cuda()
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # -> [B,1,D,H,W]

    model = TransUNet3D(
        inchannel=1,
        n_classes=4,
        base_ch=8,
        dropout=0.1,
        img_h=H,          # 用于确定 bottleneck 的 vit_img_dim = H//16
        head_num=4,
        mlp_dim=256,
        block_num=4
    ).cuda()

    y = model(x)
    print("Output:", y.shape)  # [B,4,D,H,W]

    # THOP profiling
    from thop import profile
    flops, params = profile(model, (x,), verbose=False)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
