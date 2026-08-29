# ============================================================
# Z_RiR_3D_SURROGATE.py  (FULL, RUNNABLE, NO CUDA EXTENSION)
# ------------------------------------------------------------
# ✅ 按你“路径A”的思路：彻底移除 torch.utils.cpp_extension.load + WKV CUDA
# ✅ 用纯 PyTorch LinearAttention surrogate 替代 RUN_CUDA / WKV
# ✅ 支持你的输入格式： (B, C, H, W, D) 例如 (2,3,128,128,128)
# ✅ 不再要求 D 是完全平方数（原代码 sqrt(D) 的折叠逻辑已移除）
#
# 设计说明（尽量保持你原 Z_RiR 结构）：
# - 仍然使用 Stem3D / Stage / Block / Decoder3D 的 U-Net-like 框架
# - “RWKV SpatialMix” 仍用 zigzag + q_shift，但在 3D 上采用“逐深度切片”策略：
#     outer_tokens 形状为 (B*D, H_out*W_out, C) ，每个 depth slice 作为一个 2D 序列做混合
#   这样 D=128 可直接处理，无需折叠成平方数网格。
#
# 输入 : (B,C,H,W,D)
# 输出 : (B,out_channels,H,W,D) logits   （与输入同轴顺序：H,W,D）
# ============================================================

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# timm 新接口（避免 FutureWarning）
from timm.layers import DropPath, to_2tuple, trunc_normal_


# -----------------------------
# q_shift (2D shift：输入 (B,C,H,W))
# -----------------------------
def q_shift(input, shift_pixel=1, gamma=1 / 4):
    assert gamma <= 1 / 4
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output


# ============================================================
# 1) Linear Attention surrogate (pure PyTorch)
# ============================================================
class LinearAttention(nn.Module):
    """
    Pure PyTorch linear attention surrogate (O(T·C)):
      phi(x)=elu(x)+1
      out = (phi(q) (phi(k)^T v)) / (phi(q) (phi(k)^T 1))
    q,k,v: (B,T,C)
    """
    def __init__(self, dim: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, C = q.shape
        H = self.num_heads
        d = self.head_dim

        q = q.view(B, T, H, d)
        k = k.view(B, T, H, d)
        v = v.view(B, T, H, d)

        q = self._phi(q)
        k = self._phi(k)

        kv = torch.einsum("bthd,bthe->bhde", k, v)      # (B,H,d,d)
        ksum = k.sum(dim=1)                              # (B,H,d)

        num = torch.einsum("bthd,bhde->bthe", q, kv)    # (B,T,H,d)
        den = torch.einsum("bthd,bhd->bth", q, ksum).unsqueeze(-1)  # (B,T,H,1)

        out = num / (den + self.eps)
        return out.reshape(B, T, C)


# ============================================================
# 2) VRWKV blocks (WKV -> LinearAttention)
# ============================================================
class VRWKV_SpatialMix(nn.Module):
    """
    原来：RUN_CUDA recurrence
    现在：LinearAttention recurrence（纯 PyTorch）
    仍保留：zigzag + q_shift
    """
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False,
                 scan_schemes=None, num_heads: int = 4, attn_drop: float = 0.0):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        self.recurrence = 2
        self.scan_schemes = scan_schemes or [('top-left', 'horizontal'), ('bottom-right', 'vertical')]

        # 保留结构
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.attn = LinearAttention(dim=n_embd, num_heads=num_heads)
        self.drop = nn.Dropout(attn_drop)

    def get_zigzag_indices(self, h, w, start='top-left', direction='horizontal'):
        indices = []
        if start == 'top-left':
            row_start, col_start, row_step, col_step = 0, 0, 1, 1
        elif start == 'top-right':
            row_start, col_start, row_step, col_step = 0, w - 1, 1, -1
        elif start == 'bottom-left':
            row_start, col_start, row_step, col_step = h - 1, 0, -1, 1
        elif start == 'bottom-right':
            row_start, col_start, row_step, col_step = h - 1, w - 1, -1, -1
        else:
            raise ValueError(f"Unknown start: {start}")

        if direction == 'horizontal':
            for i in range(h):
                current_row = row_start + row_step * i
                cols = list(range(w)) if (current_row % 2 == 0) else list(range(w - 1, -1, -1))
                for col in cols:
                    indices.append(current_row * w + col)
        elif direction == 'vertical':
            for i in range(w):
                current_col = col_start + col_step * i
                rows = list(range(h)) if (current_col % 2 == 0) else list(range(h - 1, -1, -1))
                for row in rows:
                    indices.append(row * w + current_col)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def jit_func(self, x, resolution, scan_scheme):
        h, w = resolution
        start, direction = scan_scheme
        zigzag_order = self.get_zigzag_indices(h, w, start=start, direction=direction)

        # x: (B,T,C), T=h*w
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x[..., zigzag_order]
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=h, w=w)

        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        return r, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        selected_scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        r, k, v = self.jit_func(x, resolution, selected_scheme)

        y1 = self.attn(q=x,  k=k, v=v)
        y1 = self.drop(y1)
        y2 = self.attn(q=y1, k=k, v=v)
        y2 = self.drop(y2)
        y = 0.5 * (y1 + y2)

        if self.key_norm is not None:
            y = self.key_norm(y)
        y = r * y
        y = self.output(y)
        return y


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)

        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.key_norm = nn.LayerNorm(hidden_sz) if key_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x


# ============================================================
# 3) Block / Stage
# ============================================================
class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, layer_id, outer_head, inner_head, num_words,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        self.has_inner = inner_dim > 0

        if self.has_inner:
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = VRWKV_SpatialMix(n_embd=inner_dim, n_layer=None, layer_id=layer_id, num_heads=max(1, inner_head))
            self.inner_norm2 = norm_layer(num_words * inner_dim)
            self.inner_ffn = VRWKV_ChannelMix(n_embd=inner_dim, n_layer=None, layer_id=layer_id)
            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = VRWKV_SpatialMix(n_embd=outer_dim, n_layer=None, layer_id=layer_id, num_heads=max(1, outer_head))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_ffn = VRWKV_ChannelMix(n_embd=outer_dim, n_layer=None, layer_id=layer_id)

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in):
        # outer_tokens: (B2, N, C) where B2 = B*D_slices
        B2, N, C = outer_tokens.size()

        if self.has_inner:
            inner_patch_resolution = [H_in, W_in]
            x = x + self.drop_path(
                self.inner_attn(
                    self.inner_norm1(x.reshape(B2, N, -1)).reshape(B2 * N, H_in * W_in, -1),
                    inner_patch_resolution
                )
            )
            x = x + self.drop_path(
                self.inner_ffn(
                    self.inner_norm2(x.reshape(B2, N, -1)).reshape(B2 * N, H_in * W_in, -1),
                    inner_patch_resolution
                )
            )
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B2, N, -1))))

        outer_patch_resolution = [H_out, W_out]
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), outer_patch_resolution))
        outer_tokens = outer_tokens + self.drop_path(self.outer_ffn(self.outer_norm2(outer_tokens), outer_patch_resolution))
        return x, outer_tokens


class Stage(nn.Module):
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1

            blocks.append(Block(
                outer_dim, _inner_dim, layer_id=j, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[j],
                act_layer=act_layer, norm_layer=norm_layer, se=se, sr_ratio=sr_ratio
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        return inner_tokens, outer_tokens


# ============================================================
# 4) Patch merging (2D)  —— 作用于每个 depth slice 的 tokens
# ============================================================
class PatchMerging2D_sentence(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=2 * stride - 1, padding=stride - 1, stride=stride)

    def forward(self, x, H, W):
        # x: (B2, N, C) with N=H*W
        B2, N, C = x.shape
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B2, C, H, W)
        x = self.conv(x)
        H2, W2 = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x = x.reshape(B2, -1, H2 * W2).transpose(1, 2)
        return x, H2, W2


class PatchMerging2D_word(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=2 * stride - 1, padding=stride - 1, stride=stride)

    def forward(self, x, H_out, W_out, H_in, W_in):
        # x: (B2*N, M, C)
        B2N, M, C = x.shape
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)

        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)

        x1 = x[:, 0::2, 0::2, :, :, :]
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4)

        x = x.reshape(-1, 2 * H_in, 2 * W_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x


# ============================================================
# 5) Stem3D / Upsample3D
# ============================================================
class Stem3D(nn.Module):
    """
    输入:  (B,C,D,H,W)
    输出:
      inner_tokens: (B2*N, M, inner_dim)  其中 B2=B*D, N=H_out*W_out, M=H_in*W_in
      outer_tokens: (B2, N, outer_dim)
      H_out,W_out,H_in,W_in,D
    """
    def __init__(self, img_size=(128, 128), in_chans=1, outer_dim=64, inner_dim=4):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.inner_dim = inner_dim

        self.num_words = 16  # 4x4
        self.common_conv = nn.Sequential(
            nn.Conv3d(in_chans, inner_dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(inner_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv3d(inner_dim * 2, inner_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv3d(inner_dim * 2, inner_dim * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(inner_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(inner_dim * 4, inner_dim * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(inner_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(inner_dim * 8, outer_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(outer_dim),
            nn.ReLU(inplace=False),
        )

        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)

    def forward(self, x):
        # x: (B,C,D,H,W)
        B, C, D, H, W = x.shape
        x = self.common_conv(x)  # (B,2*inner, D, H/2, W/2)

        # 原设计：H_out=W_out=H//8
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4
        N = H_out * W_out

        # inner_tokens
        inner_tokens = self.inner_convs(x)               # (B,inner,D,H/2,W/2)
        inner_tokens = inner_tokens.permute(0, 2, 1, 3, 4).contiguous()  # (B,D,inner,H/2,W/2)
        inner_tokens = inner_tokens.reshape(B * D, self.inner_dim, inner_tokens.shape[-2], inner_tokens.shape[-1])
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2)  # (B*D, N, inner*16)
        inner_tokens = inner_tokens.reshape(B * D * N, self.inner_dim, H_in * W_in).transpose(1, 2)  # (B*D*N,16,inner)

        # outer_tokens
        outer_tokens = self.outer_convs(x)  # (B,outer,D,H/8,W/8)
        outer_tokens = outer_tokens.permute(0, 2, 3, 4, 1).contiguous()  # (B,D,H_out,W_out,outer)
        outer_tokens = outer_tokens.reshape(B * D, N, -1)  # (B*D, N, outer_dim)

        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in), D


class UpsampleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        self.gelu1 = nn.GELU()
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x


# ============================================================
# 6) PyramidRiR_enc (3D backbone)
# ============================================================
class PyramidRiR_enc(nn.Module):
    def __init__(self, img_size=(128, 128), outer_dims=None, in_chans=1,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        depths = [2, 4, 9, 2]
        inner_dims = [4, 8, 16, 32]          # 保持你原来的设置
        outer_heads = [2, 4, 8, 16]
        inner_heads = [1, 2, 4, 8]
        sr_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed = Stem3D(img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])

        depth = 0
        self.word_merges = nn.ModuleList([])
        self.sentence_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])

        for i in range(4):
            if i > 0:
                self.word_merges.append(PatchMerging2D_word(inner_dims[i - 1], inner_dims[i]))
                self.sentence_merges.append(PatchMerging2D_sentence(outer_dims[i - 1], outer_dims[i]))

            self.stages.append(Stage(
                depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                outer_head=outer_heads[i], inner_head=inner_heads[i],
                num_patches=None, num_words=self.patch_embed.num_words,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[depth:depth + depths[i]], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i]
            ))
            depth += depths[i]

        # encoder 里做一次上采样，使得 t1~t4 适配 decoder 的 concat
        self.up_blocks = nn.ModuleList([UpsampleBlock3D(outer_dims[i], outer_dims[i]) for i in range(4)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            # 粗略fan_out初始化
            k = 1
            for kk in m.kernel_size:
                k *= kk
            fan_out = k * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: (B,C,D,H,W)
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in), D = self.patch_embed(x)
        outputs = []

        # outer_tokens: (B*D, H_out*W_out, C)
        # inner_tokens: (B*D*H_out*W_out, 16, inner_dim)
        for i in range(4):
            if i > 0:
                inner_tokens = self.word_merges[i - 1](inner_tokens, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out = self.sentence_merges[i - 1](outer_tokens, H_out, W_out)

            inner_tokens, outer_tokens = self.stages[i](inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)

            # outer_tokens -> 3D feature map: (B, C, D, H_out, W_out)
            BD, N, Cc = outer_tokens.shape
            assert BD % D == 0
            B = BD // D
            feat = outer_tokens.reshape(B, D, H_out, W_out, Cc).permute(0, 4, 1, 2, 3).contiguous()

            # encoder 内部上采样一次（只放大H/W，不动D）
            feat = self.up_blocks[i](feat)
            outputs.append(feat)

        return outputs

    def forward(self, x):
        return self.forward_features(x)


# ============================================================
# 7) Decoder3D & Final model
# ============================================================
class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_bn_relu = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x


class Z_RiR(nn.Module):
    """
    输入 : (B,C,H,W,D)
    输出 : (B,out_channels,H,W,D)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 img_size: Tuple[int, int] = (128, 128),
                 do_ds: bool = True):
        super().__init__()
        self.do_ds = do_ds
        self.num_classes = out_channels

        channels = [64, 128, 256, 512]
        self.RiR_backbone = PyramidRiR_enc(img_size=img_size, outer_dims=channels, in_chans=in_channels)

        self.decode4 = Decoder3D(channels[3], channels[2])
        self.decode3 = Decoder3D(channels[2], channels[1])
        self.decode2 = Decoder3D(channels[1], channels[0])

        # 最后把 (H/4,W/4) -> (H,W)，D 不变
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=True),
            nn.Conv3d(channels[0], out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x: (B,C,H,W,D) -> 转成 Conv3d 习惯 (B,C,D,H,W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B,C,D,H,W)

        outputs = self.RiR_backbone(x)
        t1, t2, t3, t4 = outputs[0], outputs[1], outputs[2], outputs[3]

        d4 = self.decode4(t4, t3)
        d3 = self.decode3(d4, t2)
        d2 = self.decode2(d3, t1)
        out = self.decode0(d2)  # (B,out_channels,D,H,W)

        # 输出还原为 (B,out_channels,H,W,D)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out


# ============================================================
# 8) Running example (你的数据格式)
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 你的数据格式：B,C,H,W,D
    B, C, H, W, D = 2, 3, 128, 128, 128
    x = torch.randn(B, C, H, W, D, device=device)

    model = Z_RiR(in_channels=C, out_channels=4, img_size=(H, W)).to(device).eval()

    with torch.no_grad():
        y = model(x)

    print("Device :", device)
    print("Input  :", x.shape, x.dtype)           # (B,C,H,W,D)
    print("Output :", y.shape, y.dtype, "(logits)")  # (B,out,H,W,D)

    # softmax / argmax 示例
    prob = torch.softmax(y, dim=1)
    pred = torch.argmax(prob, dim=1)  # (B,H,W,D)
    print("Prob   :", prob.shape)
    print("Pred   :", pred.shape)
