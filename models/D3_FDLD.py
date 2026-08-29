

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelNorm3D(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B,H,W,D,C]
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,H,W,D]
        return x

class DeformConv3d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("DeformConv3d currently requires an odd kernel_size.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.K = self.kernel_size ** 3

        # Equivalent to a Conv3d kernel flattened over its k^3 sampling points.
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.K)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Fixed regular k x k x k offsets in (h, w, d) order.
        coords = []
        p = self.padding
        for dh in range(-p, p + 1):
            for dw in range(-p, p + 1):
                for dd in range(-p, p + 1):
                    coords.append((dh, dw, dd))
        kernel_offsets = torch.tensor(coords, dtype=torch.float32)  # [K,3]
        self.register_buffer("kernel_offsets", kernel_offsets, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Similar fan-in scaling to Conv3d.
        fan_in = self.in_channels * self.K
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _normalize_coord(coord: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coord)
        return 2.0 * coord / float(size - 1) - 1.0

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"x must be [B,C,H,W,D], got {tuple(x.shape)}")

        B, C, H, W, D = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C}."
            )

        expected_off_c = 3 * self.K
        if offset.shape != (B, expected_off_c, H, W, D):
            raise ValueError(
                "offset must have shape "
                f"[B,{expected_off_c},H,W,D], got {tuple(offset.shape)}"
            )

        # [B, 3K, H, W, D] -> [B, K, 3, H, W, D]
        off = offset.view(B, self.K, 3, H, W, D)

        dtype = x.dtype
        device = x.device

        # Base output locations in the external [H,W,D] convention.
        h = torch.arange(H, device=device, dtype=dtype)
        w = torch.arange(W, device=device, dtype=dtype)
        d = torch.arange(D, device=device, dtype=dtype)
        hh, ww, dd = torch.meshgrid(h, w, d, indexing="ij")  # each [H,W,D]

        hh = hh.view(1, 1, H, W, D)
        ww = ww.view(1, 1, H, W, D)
        dd = dd.view(1, 1, H, W, D)

        k_off = self.kernel_offsets.to(device=device, dtype=dtype)
        kh = k_off[:, 0].view(1, self.K, 1, 1, 1)
        kw = k_off[:, 1].view(1, self.K, 1, 1, 1)
        kd = k_off[:, 2].view(1, self.K, 1, 1, 1)

        sample_h = hh + kh + off[:, :, 0, ...]
        sample_w = ww + kw + off[:, :, 1, ...]
        sample_d = dd + kd + off[:, :, 2, ...]

        gx = self._normalize_coord(sample_w, W)
        gy = self._normalize_coord(sample_h, H)
        gz = self._normalize_coord(sample_d, D)

        grid = torch.stack([gx, gy, gz], dim=-1)
        grid = grid.permute(0, 1, 4, 2, 3, 5).contiguous()
        grid = grid.view(B * self.K, D, H, W, 3)

        x_std = x.permute(0, 1, 4, 2, 3).contiguous()  # [B,C,D,H,W]
        x_rep = x_std.unsqueeze(1).expand(
            B, self.K, C, D, H, W
        ).reshape(B * self.K, C, D, H, W)

        sampled = F.grid_sample(
            x_rep,
            grid,
            mode="bilinear",       # trilinear for 5D tensors
            padding_mode="zeros",
            align_corners=True,
        )  # [B*K,C,D,H,W]

        # [B*K,C,D,H,W] -> [B,C,K,H,W,D]
        sampled = sampled.view(B, self.K, C, D, H, W)
        sampled = sampled.permute(0, 2, 1, 4, 5, 3).contiguous()

        out = torch.einsum("ock,bckhwd->bohwd", self.weight, sampled)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        return out


class LiteSpatialBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=pad,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = ChannelNorm3D(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(self.norm(x))
        return x


class ChannelFilter3D_ChannelFFT(nn.Module):

    def __init__(self,
                 channels: int,
                 mask_ratio: float = 0.75,
                 pool_factor: int = 4,
                 bias: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.mask_ratio = float(mask_ratio)
        self.pool_factor = int(pool_factor)  # compatibility only

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.freq_real = nn.Linear(self.channels, self.channels, bias=True)
        self.freq_imag = nn.Linear(self.channels, self.channels, bias=True)

        self.energy_to_score = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, self.channels),
        )

        # Optional learnable score bias, retained for API compatibility.
        if bias:
            self.score_bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter("score_bias", None)

    def compute_channel_score(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError(f"x must be [B,C,H,W,D], got {tuple(x.shape)}")

        B, C, _, _, _ = x.shape
        if C != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {C}."
            )

        # [B,C,H,W,D] -> [B,C]
        channel_vec = self.gap(x).view(B, C)

        # True 1D FFT along the channel dimension C.
        channel_fft = torch.fft.fft(
            channel_vec,
            dim=1,
            norm="ortho",
        )

        freq_real = self.freq_real(channel_fft.real)
        freq_imag = self.freq_imag(channel_fft.imag)

        freq_energy = freq_real.square() + freq_imag.square()

        score = self.energy_to_score(freq_energy)
        if self.score_bias is not None:
            score = score + self.score_bias.view(1, -1)
        score = torch.sigmoid(score)

        return score, freq_energy

    def select_channels(self, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        len_keep = max(1, int(C * (1.0 - self.mask_ratio)))

        ids = torch.topk(
            score,
            k=len_keep,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        x_flat = x.reshape(B, C, -1)
        gather_idx = ids.unsqueeze(-1).expand(-1, -1, x_flat.shape[-1])
        x_keep = torch.gather(x_flat, dim=1, index=gather_idx)
        x_keep = x_keep.view(B, len_keep, H, W, D)
        return x_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score, _ = self.compute_channel_score(x)
        return self.select_channels(x, score)


class SFAC(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 deform_ratio: float = 0.25,
                 bias: bool = True,
                 share_offset: bool = True):
        super().__init__()
        self.k = int(kernel_size)
        self.in_channels = int(in_channels)
        self.K = self.k ** 3
        self.share_offset = bool(share_offset)

        self.deform_c = max(1, int(self.in_channels * float(deform_ratio)))
        self.pass_c = self.in_channels - self.deform_c

        # TRUE 3D offsets: 3 coordinates for each of k^3 sampling points.
        offset_channels = 3 * self.K

        if self.share_offset:
            self.offset = nn.Conv3d(
                self.in_channels,
                offset_channels,
                kernel_size=self.k,
                padding=self.k // 2,
                bias=True,
            )
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        else:
            self.offset_real = nn.Conv3d(
                self.in_channels,
                offset_channels,
                kernel_size=self.k,
                padding=self.k // 2,
                bias=True,
            )
            self.offset_imag = nn.Conv3d(
                self.in_channels,
                offset_channels,
                kernel_size=self.k,
                padding=self.k // 2,
                bias=True,
            )
            nn.init.zeros_(self.offset_real.weight)
            nn.init.zeros_(self.offset_real.bias)
            nn.init.zeros_(self.offset_imag.weight)
            nn.init.zeros_(self.offset_imag.bias)

        self.deform_real = DeformConv3d(
            self.deform_c,
            self.deform_c,
            kernel_size=self.k,
            bias=bias,
        )
        self.deform_imag = DeformConv3d(
            self.deform_c,
            self.deform_c,
            kernel_size=self.k,
            bias=bias,
        )

        self.att_pool = nn.AdaptiveAvgPool3d(1)
        self.att_conv = nn.Conv3d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True,
        )
        self.res = nn.Conv3d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True,
        )

        self.norm = ChannelNorm3D(self.in_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        res = self.res(x)

        att = self.att_conv(x)
        att = self.att_pool(att)

        # Genuine 3D FFT over all three spatial dimensions.
        x_fft = torch.fft.fftn(
            res,
            s=(H, W, D),
            dim=(-3, -2, -1),
        )

        x_real = x_fft.real
        x_imag = x_fft.imag

        xd_real = x_real[:, :self.deform_c, ...]
        xd_imag = x_imag[:, :self.deform_c, ...]
        xp_real = x_real[:, self.deform_c:, ...]
        xp_imag = x_imag[:, self.deform_c:, ...]

        if self.share_offset:
            # Preserve the original Lite design: one 3D offset field shared by
            # the real and imaginary branches.
            off = self.offset(res)
            off_real = off
            off_imag = off
        else:
            # Closer to the manuscript formulation: independent 3D offsets.
            off_real = self.offset_real(x_real)
            off_imag = self.offset_imag(x_imag)

        real_f = self.deform_real(xd_real, off_real)
        imag_f = self.deform_imag(xd_imag, off_imag)

        if self.pass_c > 0:
            real_all = torch.cat([real_f, xp_real], dim=1)
            imag_all = torch.cat([imag_f, xp_imag], dim=1)
        else:
            real_all, imag_all = real_f, imag_f

        x_ifft_in = real_all + 1j * imag_all
        x_out = torch.fft.ifftn(
            x_ifft_in,
            s=(H, W, D),
            dim=(-3, -2, -1),
        ).abs()

        out = self.norm(self.act(x_out * att + res))
        return out


class CFAE(nn.Module):
    def __init__(self, in_channels, out_channels, mask_ratio=0.75, pool_factor=4):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        keep_channels = max(1, int(out_channels * (1 - self.mask_ratio)))

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            ChannelNorm3D(out_channels),
            nn.GELU(),
            ChannelFilter3D_ChannelFFT(
                channels=out_channels,
                mask_ratio=self.mask_ratio,
                pool_factor=pool_factor,
                bias=False,
            ),
            ChannelNorm3D(keep_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    """Downsample by x4 in one step: 128->32->8->2."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck_fftdeform=False,
                 down_factor=4,
                 kernel_size=3,
                 deform_ratio=0.25,
                 share_offset=True):
        super().__init__()
        assert down_factor in (2, 4)
        self.pool = nn.MaxPool3d(
            kernel_size=down_factor,
            stride=down_factor,
            padding=0,
        )

        if use_bottleneck_fftdeform:
            self.spatial = nn.Sequential(
                SFAC(
                    in_channels,
                    kernel_size=kernel_size,
                    deform_ratio=deform_ratio,
                    share_offset=share_offset,
                ),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
                ChannelNorm3D(out_channels),
                nn.GELU(),
            )
        else:
            self.spatial = LiteSpatialBlock3D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
            )

    def forward(self, x):
        x = self.pool(x)
        x = self.spatial(x)
        return x


class Up3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_factor=4,
                 trilinear=True,
                 kernel_size=3,
                 use_fftdeform=False,
                 deform_ratio=0.25,
                 share_offset=True):
        super().__init__()
        assert up_factor in (2, 4)
        self.use_fftdeform = bool(use_fftdeform)

        if trilinear:
            self.up = nn.Upsample(
                scale_factor=up_factor,
                mode="trilinear",
                align_corners=True,
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=up_factor,
                stride=up_factor,
            )

        if self.use_fftdeform:
            self.conv = nn.Sequential(
                SFAC(
                    in_channels,
                    kernel_size=kernel_size,
                    deform_ratio=deform_ratio,
                    share_offset=share_offset,
                ),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
                ChannelNorm3D(out_channels),
                nn.GELU(),
            )
        else:
            self.conv = LiteSpatialBlock3D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffH = x2.size(2) - x1.size(2)
        diffW = x2.size(3) - x1.size(3)
        diffD = x2.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            [
                diffD // 2, diffD - diffD // 2,
                diffW // 2, diffW - diffW // 2,
                diffH // 2, diffH - diffH // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(True),  # preserved from the original implementation
        )

    def forward(self, x):
        return self.net(x)


class FDLD(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 base=64,
                 c1=24,
                 c2=48,
                 c3=96,
                 mask_ratio=0.75,
                 tokenfft_pool_factor=4,
                 deform_ratio=0.25,
                 bilinear=True,
                 share_offset=True,
                 decoder_fftdeform=(True, False, False)):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        keep = lambda c: max(1, int(c * (1 - self.mask_ratio)))
        self.decoder_fftdeform = decoder_fftdeform

        self.inc = CFAE(
            n_channels,
            base,
            mask_ratio=self.mask_ratio,
            pool_factor=tokenfft_pool_factor,
        )

        self.down1 = Down3D(
            keep(base), c1,
            use_bottleneck_fftdeform=False,
            down_factor=4,
        )
        self.down2 = Down3D(
            c1, c2,
            use_bottleneck_fftdeform=False,
            down_factor=4,
        )
        self.down3 = Down3D(
            c2, c3,
            use_bottleneck_fftdeform=True,
            down_factor=4,
            deform_ratio=deform_ratio,
            share_offset=share_offset,
        )

        factor = 2 if bilinear else 1

        self.up2 = Up3D(
            c3 + c2,
            c2 // factor,
            up_factor=4,
            trilinear=bilinear,
            use_fftdeform=self.decoder_fftdeform[0],
            deform_ratio=deform_ratio,
            share_offset=share_offset,
        )
        self.up3 = Up3D(
            (c2 // factor) + c1,
            max(8, c1 // factor),
            up_factor=4,
            trilinear=bilinear,
            use_fftdeform=self.decoder_fftdeform[1],
            deform_ratio=deform_ratio,
            share_offset=share_offset,
        )
        self.up4 = Up3D(
            max(8, c1 // factor) + keep(base),
            64,
            up_factor=4,
            trilinear=bilinear,
            use_fftdeform=self.decoder_fftdeform[2],
            deform_ratio=deform_ratio,
            share_offset=share_offset,
        )

        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 128, 128).cuda()

    model = FDLD(
        n_channels=3,
        n_classes=4,
        base=64,
        c1=24, c2=48, c3=96,
        mask_ratio=0.75,
        tokenfft_pool_factor=4,
        deform_ratio=0.25,
        bilinear=True,
        decoder_fftdeform=(True, False, False)
    ).cuda()

    y = model(x)
    print("output:", y.shape)  # [2, 2, 128, 128, 128]

