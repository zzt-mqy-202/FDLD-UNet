import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models import register_model
from torchvision.ops.deform_conv import deform_conv2d

__all__ = ['FDLD_UNet',]


class filter_cnn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, mask_ratio=0.75) -> None:
        super(filter_cnn, self).__init__()
        self.filter_cnn2d = A_SpatialFilter(int(out_channels*(1-mask_ratio)), kernel_size, image_size)
        self.filter_cnn1d = ChannelFilter(in_channels, out_channels, image_size, mask_ratio=mask_ratio)

    def forward(self, x) -> torch.Tensor:
        x = self.filter_cnn1d(x)
        return self.filter_cnn2d(x)


class ChannelFilter(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, hidden_dim=None, bias=True, mask_ratio=0.75,) -> None:
        super(ChannelFilter, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim or 2 * out_channels
        self.out_channels = out_channels
        self.weight1 = nn.Parameter(torch.randn(1, self.hidden_dim, 1, 1, 2, dtype=torch.float32) * 0.02)
        self.norm1 = ChannelNorm(self.hidden_dim)
        self.weight2 = nn.Parameter(torch.randn(1, out_channels, 1, 1, 2, dtype=torch.float32) * 0.02)
        self.norm2 = ChannelNorm(out_channels)
        self.active = nn.GELU()
        self.bias = bias and nn.Parameter(torch.randn(out_channels))
        self.mask_ratio = mask_ratio
        self.masked_L = nn.Sequential(
            nn.AvgPool2d(image_size),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def random_masking(self, x, mask_ratio=0.75):
        B, C, H, W, = x.shape
        len_keep = int(C * (1 - mask_ratio))

        noise = self.masked_L(x)
        noise = torch.sigmoid(noise)
        noise = noise.reshape(shape=(x.shape[0], -1))
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x.reshape(B, C, -1), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, H * W))

        mask = torch.ones([B, C], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked.reshape(B, len_keep, H, W), mask, ids_restore

    def forward(self, x) -> torch.Tensor:
        weight1 = torch.view_as_complex(self.weight1)
        weight2 = torch.view_as_complex(self.weight2)

        x = torch.fft.fft(x, n=self.hidden_dim, dim=1, norm='ortho')
        x = x * weight1
        x = torch.fft.ifft(x, dim=1, norm='ortho').abs()
        x = self.norm1(x)

        x = torch.fft.fft(x, n=self.out_channels, dim=1, norm='ortho')
        x = x * weight2
        x = torch.fft.ifft(x, dim=1, norm='ortho').abs()
        x = self.norm2(x)

        if isinstance(self.bias, torch.Tensor):
            x = x + self.bias.unsqueeze(-1).unsqueeze(-1)

        x = self.active(x)
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        return x


class ChannelNorm(nn.Module):
    def __init__(self, in_channels: int):
        super(ChannelNorm, self).__init__()
        self.norm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(in_channels),
            Rearrange('b h w c -> b c h w'),
        )

    def forward(self, x):
        return self.norm(x)


class A_SpatialFilter(nn.Module):
    def __init__(self, in_channels, kernel_size, image_size,  bias=True):
        super(A_SpatialFilter, self).__init__()
        self.k_size = kernel_size
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, kernel_size, kernel_size))
        self.offseti = nn.Parameter(torch.randn(in_channels, image_size, image_size))
        self.offsetj = nn.Parameter(torch.randn(in_channels, image_size, image_size))
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2 * kernel_size * kernel_size, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = ChannelNorm(in_channels)
        self.bias = bias and nn.Parameter(torch.randn(in_channels))

    def gen_offset(self, x, k_size):
        C, H, W = x.shape
        x = self.conv1(x.reshape(1, C, H, W))
        x = x - torch.mean(x)
        return torch.sigmoid(x)*k_size

    def forward(self, x):
        B, C, H, W = x.shape
        k_size = self.k_size
        offseti = self.gen_offset(self.offseti, k_size)
        offsetj = self.gen_offset(self.offsetj, k_size)
        weight = torch.fft.fft2(self.weight, s=(k_size, k_size), dim=(-2, -1))
        x = torch.fft.fft2(x, s=(H, W), dim=(-2, -1))
        x_real = deform_conv2d(x.real, offseti.expand(B, -1, H, W), weight.real, padding=(k_size//2, k_size//2))
        x_imag = deform_conv2d(x.imag, offsetj.expand(B, -1, H, W), weight.imag, padding=(k_size//2, k_size//2))
        x = x_real+1j*x_imag
        x = torch.fft.ifft2(x, s=(H, W), dim=(-2, -1)).abs()
        x = self.norm(x)

        if isinstance(self.bias, torch.Tensor):
            x = x + self.bias.unsqueeze(-1).unsqueeze(-1)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, imgsize=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            filter_cnn(in_channels, out_channels, 3, imgsize),
            nn.MaxPool2d(5, 4, 2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, imgsize=256):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.conv = filter_cnn(in_channels, out_channels, 3, imgsize)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=4)
            self.conv = filter_cnn(in_channels, out_channels, 3, imgsize)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FDLD_UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 224,
        window_size: int | None = None,
        bilinear=True,
        **kwargs
    ) -> None:
        super(FDLD_UNet, self).__init__()
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.window = window_size or img_size

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 128, img_size)
        self.down2 = Down(int(128*0.25), 256, img_size//4)
        self.down3 = Down(int(256*0.25), 512, img_size//16)
        factor = 2 if bilinear else 1
        self.down4 = Down(int(512*0.25), 1024 // factor, img_size//64)
        self.up1 = Up(int(1024*0.25), 512 // factor, bilinear, img_size//64)
        self.up2 = Up(int(512*0.25), 256 // factor, bilinear, img_size//16)
        self.up3 = Up(int(256*0.25), 128 // factor, bilinear, img_size//4)
        self.up4 = Up(48, 64, bilinear, img_size)
        self.outConv = OutConv(int(64*0.25), self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outConv(x)

        return logits


@register_model
def fdld_unet(**kwargs) -> FDLD_UNet:
    model = FDLD_UNet(**kwargs)
    return model