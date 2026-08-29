import torch
from torch import nn
import torch.nn.functional as F


def match_size_3d(x, ref):
    """
    让 x 的 (D,H,W) 与 ref 完全一致：
    - 若 x 比 ref 小：对 x 做对称 padding
    - 若 x 比 ref 大：对 x 做中心裁剪
    x/ref: [B, C, D, H, W]
    """
    _, _, d, h, w = x.shape
    _, _, rd, rh, rw = ref.shape

    # ----- Pad if needed -----
    pd = max(rd - d, 0)
    ph = max(rh - h, 0)
    pw = max(rw - w, 0)

    if pd > 0 or ph > 0 or pw > 0:
        # F.pad 的顺序是 (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(
            x,
            (pw // 2, pw - pw // 2,
             ph // 2, ph - ph // 2,
             pd // 2, pd - pd // 2)
        )
        _, _, d, h, w = x.shape

    # ----- Center crop if needed -----
    sd = max((d - rd) // 2, 0)
    sh = max((h - rh) // 2, 0)
    sw = max((w - rw) // 2, 0)
    x = x[:, :, sd:sd + rd, sh:sh + rh, sw:sw + rw]
    return x


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.Dropout3d(p) if p > 0 else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NestedUNet3D(nn.Module):
    """
    3D UNet++ (Nested U-Net) - anisotropic down/up sampling:
    只在 H/W 做 2 倍下采样，上采样；D 维保持不变（更适合 D 较小的体数据）。
    输入:  [B, in_channel, D, H, W]
    输出:  [B, out_channel, D, H, W] (logits)
    """
    def __init__(self, in_channel: int, out_channel: int, base_ch: int = 4, dropout: float = 0.0):
        super().__init__()

        nb_filter = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        # ✅ 关键修正：只对 H/W pooling，不动 D
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ✅ 关键修正：只对 H/W upsample，不动 D
        self.up = lambda x: F.interpolate(x, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)

        self.conv0_0 = DoubleConv3D(in_channel, nb_filter[0], p=dropout)
        self.conv1_0 = DoubleConv3D(nb_filter[0], nb_filter[1], p=dropout)
        self.conv2_0 = DoubleConv3D(nb_filter[1], nb_filter[2], p=dropout)
        self.conv3_0 = DoubleConv3D(nb_filter[2], nb_filter[3], p=dropout)
        self.conv4_0 = DoubleConv3D(nb_filter[3], nb_filter[4], p=dropout)

        self.conv0_1 = DoubleConv3D(nb_filter[0] + nb_filter[1], nb_filter[0], p=dropout)
        self.conv1_1 = DoubleConv3D(nb_filter[1] + nb_filter[2], nb_filter[1], p=dropout)
        self.conv2_1 = DoubleConv3D(nb_filter[2] + nb_filter[3], nb_filter[2], p=dropout)
        self.conv3_1 = DoubleConv3D(nb_filter[3] + nb_filter[4], nb_filter[3], p=dropout)

        self.conv0_2 = DoubleConv3D(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], p=dropout)
        self.conv1_2 = DoubleConv3D(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], p=dropout)
        self.conv2_2 = DoubleConv3D(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], p=dropout)

        self.conv0_3 = DoubleConv3D(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], p=dropout)
        self.conv1_3 = DoubleConv3D(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], p=dropout)

        self.conv0_4 = DoubleConv3D(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], p=dropout)

        self.final = nn.Conv3d(nb_filter[0], out_channel, kernel_size=1)

    def _up_to(self, x, ref):
        x = self.up(x)
        if x.shape[2:] != ref.shape[2:]:
            x = match_size_3d(x, ref)
        return x

    def forward(self, x):
        # x: [B,C,D,H,W]
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_to(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_to(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_to(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_to(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_to(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_to(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_to(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_to(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_to(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_to(x1_3, x0_0)], dim=1))

        out = self.final(x0_4)  # logits: [B, out_channel, D, H, W]
        return out


if __name__ == "__main__":
    # 例子：D 可以很小也不会被 pool 掉（因为我们不下采样 D）
    B, C, D, H, W = 2, 1, 128, 128, 128
    x = torch.randn(B, C, D, H, W).cuda()

    net = NestedUNet3D(in_channel=1, out_channel=2, base_ch=4, dropout=0.0).cuda()
    y = net(x)
    print("out:", y.shape)

    from thop import profile
    flops, params = profile(net, (x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
