import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# =========================
# Utils
# =========================
def _to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    assert len(x) == 3
    return tuple(x)

def q_shift3d(x: torch.Tensor, shift: int, D: int, H: int, W: int) -> torch.Tensor:
    """
    3D shift-mix:
      split channels into 6 chunks and shift +/-D, +/-H, +/-W.
    x: (B, T, C) with T = D*H*W
    """
    B, T, C = x.shape
    assert T == D * H * W
    if shift <= 0:
        return x

    x3d = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
    out = x3d.clone()

    c6 = C // 6
    c_end = 6 * c6

    if c6 > 0:
        # +D
        out[:, 0:c6, :-shift, :, :] = x3d[:, 0:c6, shift:, :, :]
        out[:, 0:c6, -shift:, :, :] = 0
        # -D
        out[:, c6:2*c6, shift:, :, :] = x3d[:, c6:2*c6, :-shift, :, :]
        out[:, c6:2*c6, :shift, :, :] = 0
        # +H
        out[:, 2*c6:3*c6, :, :-shift, :] = x3d[:, 2*c6:3*c6, :, shift:, :]
        out[:, 2*c6:3*c6, :, -shift:, :] = 0
        # -H
        out[:, 3*c6:4*c6, :, shift:, :] = x3d[:, 3*c6:4*c6, :, :-shift, :]
        out[:, 3*c6:4*c6, :, :shift, :] = 0
        # +W
        out[:, 4*c6:5*c6, :, :, :-shift] = x3d[:, 4*c6:5*c6, :, :, shift:]
        out[:, 4*c6:5*c6, :, :, -shift:] = 0
        # -W
        out[:, 5*c6:c_end, :, :, shift:] = x3d[:, 5*c6:c_end, :, :, :-shift]
        out[:, 5*c6:c_end, :, :, :shift] = 0

    if c_end < C:
        out[:, c_end:, :, :, :] = x3d[:, c_end:, :, :, :]

    return out.view(B, C, T).transpose(1, 2).contiguous()

# =========================
# Linear Attention surrogate
# =========================
class LinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, C = q.shape
        Hh, d = self.num_heads, self.head_dim
        q = self._phi(q.view(B, T, Hh, d))
        k = self._phi(k.view(B, T, Hh, d))
        v = v.view(B, T, Hh, d)

        kv = torch.einsum("bthd,bthe->bhde", k, v)  # (B,Hh,d,d)
        ksum = k.sum(dim=1)                          # (B,Hh,d)
        num = torch.einsum("bthd,bhde->bthe", q, kv)
        den = torch.einsum("bthd,bhd->bth", q, ksum).unsqueeze(-1)
        out = num / (den + self.eps)
        return out.reshape(B, T, C)

# =========================
# HFE 3D
# =========================
class Separation3D(nn.Module):
    def __init__(self, pool_kernel: int = 2):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, x: torch.Tensor):
        fl = self.pool(x)
        fl_up = F.interpolate(fl, size=x.shape[-3:], mode="trilinear", align_corners=False)
        fh = x - fl_up
        return fh, fl

class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.fc(self.avg(x))

class Integration3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw_low  = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_high = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw_fuse = nn.Conv3d(2 * channels, channels, 1, bias=False)

        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.ca = ChannelAttention3D(channels)
        self.out = nn.Conv3d(channels, channels, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, fl_up: torch.Tensor, fh: torch.Tensor):
        fl = self.act(self.dw_low(fl_up))
        fh = self.act(self.dw_high(fh))
        x = torch.cat([fl, fh], dim=1)
        x = self.act(self.pw_fuse(x))
        x = self.act(self.conv(x))
        x = self.ca(x)
        return self.out(x)

# =========================
# RWKV-like blocks
# =========================
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, shift_size: int = 1, drop: float = 0.0):
        super().__init__()
        self.shift_size = shift_size
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.attn = LinearAttention(dim, num_heads=num_heads)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        xs = q_shift3d(x, self.shift_size, D, H, W)
        k = self.key(xs)
        v = self.value(xs)
        r = torch.sigmoid(self.receptance(xs))
        y = self.attn(q=x, k=k, v=v)
        y = self.drop(y)
        return self.output(r * y)

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 4.0, shift_size: int = 1, drop: float = 0.0):
        super().__init__()
        self.shift_size = shift_size
        hidden = int(dim * hidden_ratio)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        xs = q_shift3d(x, self.shift_size, D, H, W)
        k = F.relu(self.key(xs), inplace=True) ** 2
        v = self.value(k)
        r = torch.sigmoid(self.receptance(xs))
        return self.drop(r * v)

class DownBlock_HFE_RWKV_3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, shift_size: int = 1, pool_kernel: int = 2, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.sep = Separation3D(pool_kernel=pool_kernel)
        self.intg = Integration3D(dim)
        self.spatial = VRWKV_SpatialMix(dim, num_heads=num_heads, shift_size=shift_size, drop=drop)
        self.channel = VRWKV_ChannelMix(dim, hidden_ratio=4.0, shift_size=shift_size, drop=drop)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        B, T, C = x.shape
        x_map = x.transpose(1, 2).contiguous().view(B, C, D, H, W)

        fh, fl = self.sep(x_map)

        fh_tok = fh.view(B, C, -1).transpose(1, 2).contiguous()
        fh_tok = fh_tok + self.spatial(self.norm1(fh_tok), D, H, W)
        fh_map = fh_tok.transpose(1, 2).contiguous().view(B, C, D, H, W)

        fl_up = F.interpolate(fl, size=(D, H, W), mode="trilinear", align_corners=False)
        x_int = self.intg(fl_up, fh_map)

        x_tok = x_int.view(B, C, -1).transpose(1, 2).contiguous()
        x_tok = x_tok + self.channel(self.norm2(x_tok), D, H, W)
        return x_tok

# =========================
# Patch ops 3D
# =========================
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        p = _to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=p, stride=p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (B,embed,D',H',W')
        D, H, W = x.shape[-3:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, D, H, W

class PatchMerging3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        B, T, C = x.shape
        x = x.view(B, D, H, W, C)

        pd, ph, pw = D % 2, H % 2, W % 2
        if pd or ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
            D, H, W = x.shape[1], x.shape[2], x.shape[3]

        x000 = x[:, 0::2, 0::2, 0::2, :]
        x001 = x[:, 0::2, 0::2, 1::2, :]
        x010 = x[:, 0::2, 1::2, 0::2, :]
        x011 = x[:, 0::2, 1::2, 1::2, :]
        x100 = x[:, 1::2, 0::2, 0::2, :]
        x101 = x[:, 1::2, 0::2, 1::2, :]
        x110 = x[:, 1::2, 1::2, 0::2, :]
        x111 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x000, x001, x010, x011, x100, x101, x110, x111], dim=-1)  # (B,D/2,H/2,W/2,8C)
        D2, H2, W2 = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, D2 * H2 * W2, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, D2, H2, W2

class PatchExpand3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        B, T, C = x.shape
        x = self.expand(x)  # (B,T,4C)
        x = x.view(B, D, H, W, 4 * C)

        assert (4 * C) % 8 == 0
        C2 = (4 * C) // 8  # = C/2

        x = x.view(B, D, H, W, 2, 2, 2, C2)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D * 2, H * 2, W * 2, C2)

        D2, H2, W2 = D * 2, H * 2, W * 2
        x = x.view(B, D2 * H2 * W2, C2)
        x = self.norm(x)
        return x, D2, H2, W2

class FinalUpsampleX4_3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up1 = PatchExpand3D(dim)
        self.up2 = PatchExpand3D(dim // 2)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        x, D, H, W = self.up1(x, D, H, W)
        x, D, H, W = self.up2(x, D, H, W)
        return x, D, H, W

class ResBlock3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        B, T, C = x.shape
        x_map = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        y = self.act(self.conv1(x_map))
        y = self.conv2(y)
        x_map = x_map + y
        return x_map.flatten(2).transpose(1, 2).contiguous()

# =========================
# Encoder / Decoder 3D
# =========================
class BasicLayerEncoder3D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            DownBlock_HFE_RWKV_3D(dim, num_heads=num_heads, shift_size=1, pool_kernel=2, drop=drop)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, D, H, W)
        return x

class Encoder3D(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int = 64,
        patch_size: int = 4,
        depths: List[int] = (1, 1, 1, 1),
        heads: List[int] = (2, 2, 4, 4),
        drop: float = 0.0,
    ):
        super().__init__()
        self.patch = PatchEmbed3D(in_chans, embed_dim, patch_size=patch_size)

        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.stages = nn.ModuleList([BasicLayerEncoder3D(dims[i], depths[i], heads[i], drop=drop) for i in range(4)])
        self.merges = nn.ModuleList([PatchMerging3D(dims[i]) for i in range(3)])

    def forward(self, x: torch.Tensor):
        skips, shapes = [], []
        x, D, H, W = self.patch(x)
        skips.append(x); shapes.append((D, H, W))

        for i in range(4):
            x = self.stages[i](x, D, H, W)
            if i < 3:
                x, D, H, W = self.merges[i](x, D, H, W)
                skips.append(x); shapes.append((D, H, W))

        return skips, shapes

class Decoder3D(nn.Module):
    def __init__(self, embed_dim: int = 64, depths: List[int] = (1, 1, 1)):
        super().__init__()
        d0, d1, d2, d3 = embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8

        self.up3 = PatchExpand3D(d3)  # 8E -> 4E
        self.up2 = PatchExpand3D(d2)  # 4E -> 2E
        self.up1 = PatchExpand3D(d1)  # 2E -> E

        self.ref3 = nn.ModuleList([ResBlock3D(d2) for _ in range(depths[0])])
        self.ref2 = nn.ModuleList([ResBlock3D(d1) for _ in range(depths[1])])
        self.ref1 = nn.ModuleList([ResBlock3D(d0) for _ in range(depths[2])])

        self.final_up = FinalUpsampleX4_3D(d0)

    def forward(self, skips, shapes):
        x0, x1, x2, x3 = skips[0], skips[1], skips[2], skips[3]
        (D0, H0, W0), (D1, H1, W1), (D2, H2, W2), (D3, H3, W3) = shapes

        x, D, H, W = self.up3(x3, D3, H3, W3)
        x = x + x2
        for b in self.ref3:
            x = b(x, D, H, W)

        x, D, H, W = self.up2(x, D, H, W)
        x = x + x1
        for b in self.ref2:
            x = b(x, D, H, W)

        x, D, H, W = self.up1(x, D, H, W)
        x = x + x0
        for b in self.ref1:
            x = b(x, D, H, W)

        x, D, H, W = self.final_up(x, D, H, W)
        return x, D, H, W

# =========================
# Full Model: accepts (B,C,H,W,D)
# =========================
class HFERWKVSeg3D_HWDSupport(nn.Module):
    """
    Input : (B, C, H, W, D)   <-- your data layout
    Intern: (B, C, D, H, W)
    Output: (B, num_classes, H, W, D)
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dim: int = 64,
        patch_size: int = 4,
        enc_depths: List[int] = (1, 1, 1, 1),
        enc_heads: List[int] = (2, 2, 4, 4),
        dec_depths: List[int] = (1, 1, 1),
        drop: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder3D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=list(enc_depths),
            heads=list(enc_heads),
            drop=drop,
        )
        self.decoder = Decoder3D(embed_dim=embed_dim, depths=list(dec_depths))

        # After FinalUpsampleX4_3D: channels become embed_dim/4
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.head = nn.Conv3d(embed_dim // 4, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W,D) -> (B,C,D,H,W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        skips, shapes = self.encoder(x)
        tok, D, H, W = self.decoder(skips, shapes)

        B, T, C = tok.shape
        feat = tok.transpose(1, 2).contiguous().view(B, C, D, H, W)
        logits = self.head(feat)  # (B,num_classes,D,H,W)

        # back to (B,num_classes,H,W,D)
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        return logits

# =========================
# Usage example (your shape)
# =========================
if __name__ == "__main__":
    B, C, H, W, D = 2, 3, 128, 128, 128
    x = torch.randn(B, C, H, W, D).cuda()

    model = HFERWKVSeg3D_HWDSupport(
        in_chans=C,
        num_classes=4,
        embed_dim=64,
        patch_size=4,
        enc_depths=(1, 1, 1, 1),
        enc_heads=(2, 2, 4, 4),
        dec_depths=(1, 1, 1),
    ).cuda()

    y = model(x)
    print("x:", x.shape)  # (2,3,128,128,128)
    print("y:", y.shape)  # (2,4,128,128,128)
