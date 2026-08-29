import torch
from torch import nn
import torch.nn.functional as F

def center_crop_to(x, ref):
    _, _, d, h, w = x.shape
    _, _, rd, rh, rw = ref.shape
    sd = max((d - rd) // 2, 0)
    sh = max((h - rh) // 2, 0)
    sw = max((w - rw) // 2, 0)
    return x[:, :, sd:sd+rd, sh:sh+rh, sw:sw+rw]

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.Dropout3d(p),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Up3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.reduce = nn.Conv3d(ch, ch // 2, 1, 1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.reduce(x)
        if x.shape[2:] != skip.shape[2:]:
            x = center_crop_to(x, skip)
        return torch.cat([x, skip], dim=1)

class UNet3D(nn.Module):
    """
    base_ch 建议最小=4（再小会很弱且BN不稳）
    """
    def __init__(self, inchannel=1, n_classes=4, base_ch=8, dropout=0.1):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.enc1 = ConvBlock3D(inchannel, c1, p=dropout)
        self.down1 = Down3D(c1)

        self.enc2 = ConvBlock3D(c1, c2, p=dropout)
        self.down2 = Down3D(c2)

        self.enc3 = ConvBlock3D(c2, c3, p=dropout)
        self.down3 = Down3D(c3)

        self.enc4 = ConvBlock3D(c3, c4, p=dropout)
        self.down4 = Down3D(c4)

        self.bottleneck = ConvBlock3D(c4, c5, p=dropout)

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
        r1 = self.enc1(x)
        r2 = self.enc2(self.down1(r1))
        r3 = self.enc3(self.down2(r2))
        r4 = self.enc4(self.down3(r3))
        r5 = self.bottleneck(self.down4(r4))

        o1 = self.dec1(self.up1(r5, r4))
        o2 = self.dec2(self.up2(o1, r3))
        o3 = self.dec3(self.up3(o2, r2))
        o4 = self.dec4(self.up4(o3, r1))

        return self.out(o4)  # logits [B, n_classes, D, H, W]

if __name__ == "__main__":
    # 假设 patch = (128,128,16)
    B, C, H, W, D = 2, 1, 128, 128, 128

    x = torch.randn(B, C, H, W, D).cuda()
    x = x.permute(0, 1, 4, 2, 3)   # → [B,1,D,H,W]

    model = UNet3D(inchannel=1, n_classes=4).cuda()
    y = model(x)

    print(y.shape)
    from thop import profile
    flops, params = profile(model, (x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')