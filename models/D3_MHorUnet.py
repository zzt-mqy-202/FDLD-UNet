import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath


# -------------------------
# 3D LayerNorm (channels_last / channels_first)
# -------------------------
class LayerNorm3D(nn.Module):
    """
    Supports:
      - channels_last : (B, D, H, W, C)
      - channels_first: (B, C, D, H, W)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ["channels_last", "channels_first"]
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # x: (B, D, H, W, C)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            # x: (B, C, D, H, W)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


# -------------------------
# 3D Depthwise Conv helper
# -------------------------
def get_dwconv3d(dim, kernel, bias):
    return nn.Conv3d(
        dim, dim,
        kernel_size=kernel,
        padding=(kernel - 1) // 2,
        bias=bias,
        groups=dim
    )


# -------------------------
# gnconv for 3D
# keep the same "order" decomposition logic, but using Conv3d
# -------------------------
class gnconv3d(nn.Module):
    def __init__(self, dim, order=5, s=1.0, dw_kernel=7):
        super().__init__()
        self.order = order
        # same logic: dims split
        self.dims = [dim // (2 ** i) for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv3d(dim, 2 * dim, kernel_size=1)

        self.dwconv = get_dwconv3d(sum(self.dims), dw_kernel, True)
        self.proj_out = nn.Conv3d(dim, dim, kernel_size=1)
        self.pws = nn.ModuleList(
            [nn.Conv3d(self.dims[i], self.dims[i + 1], kernel_size=1) for i in range(order - 1)]
        )
        self.scale = s

    def forward(self, x):
        # x: (B,C,D,H,W)
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)

        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)
        return x


# -------------------------
# 3D Block (Horblock-like)
# -------------------------
class Block3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv3d):
        super().__init__()
        self.norm1 = LayerNorm3D(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = LayerNorm3D(dim, eps=1e-6, data_format='channels_last')

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: (B,C,D,H,W)
        B, C, D, H, W = x.shape
        gamma1 = self.gamma1.view(C, 1, 1, 1) if self.gamma1 is not None else 1.0
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        shortcut = x
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B,D,H,W,C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,D,H,W)

        x = shortcut + self.drop_path(x)
        return x


# -------------------------
# 3D SC Attention Bridge
# -------------------------
class Channel_Att_Bridge3D(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        # t*: (B,C,D,H,W)
        att = torch.cat((
            self.avgpool(t1),
            self.avgpool(t2),
            self.avgpool(t3),
            self.avgpool(t4),
            self.avgpool(t5),
        ), dim=1)  # (B, sumC, 1,1,1)

        att = att.squeeze(-1).squeeze(-1).transpose(-1, -2)  # (B,1,sumC)
        att = self.get_all_att(att)  # (B,1,sumC)
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)

        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))

        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 3D version: use Conv3d
        self.shared_conv3d = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, stride=1, padding=9, dilation=3),
            nn.Sigmoid()
        )

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv3d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge3D(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = Channel_Att_Bridge3D(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge3D()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


# -------------------------
# MHorunet3D
# -------------------------
class MHorunet3D(nn.Module):
    """
    Input : [B, C, H, W, D]  (UNet3D demo style)
    Intern: permute -> [B, C, D, H, W]
    Output: logits [B, num_classes, D, H, W]  (UNet3D-style)
            if apply_sigmoid=True -> sigmoid probs
    """
    def __init__(
        self,
        num_classes=1,
        input_channels=1,
        layer_scale_init_value=1e-6,
        block=Block3D,
        use_checkpoint=False,
        c_list=[8, 16, 32, 64, 128, 256],
        depths=[2, 3, 18, 2],
        drop_path_rate=0.,
        split_att='fc',
        bridge=True,
        apply_sigmoid=False,
        upsample_mode="trilinear",
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.bridge = bridge
        self.apply_sigmoid = apply_sigmoid
        self.upsample_mode = upsample_mode

        # first two simple conv stages
        self.encoder1 = nn.Sequential(nn.Conv3d(input_channels, c_list[0], 3, stride=1, padding=1, bias=True))
        self.encoder2 = nn.Sequential(nn.Conv3d(c_list[0], c_list[1], 3, stride=1, padding=1, bias=True))

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # gnconv choices (3D versions)
        gnconvs = [
            partial(gnconv3d, order=2, s=1 / 3),
            partial(gnconv3d, order=3, s=1 / 3),
            partial(gnconv3d, order=4, s=1 / 3),  # 原来这里用 GlobalLocalFilter(FFT2D)，3D版为稳定性改为纯3D卷积
            partial(gnconv3d, order=5, s=1 / 3),
        ]

        self.encoder3 = nn.Sequential(
            *[block(dim=c_list[1], drop_path=dp_rates[0 + j],
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[0]) for j in range(depths[0])],
            nn.Conv3d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        self.encoder4 = nn.Sequential(
            *[block(dim=c_list[2], drop_path=dp_rates[2 + j],
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[1]) for j in range(depths[1])],
            nn.Conv3d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )

        self.encoder5 = nn.Sequential(
            *[block(dim=c_list[3], drop_path=dp_rates[5 + j],
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[2]) for j in range(depths[2])],
            nn.Conv3d(c_list[3], c_list[4], 3, stride=1, padding=1),
        )

        self.encoder6 = nn.Sequential(
            *[block(dim=c_list[4], drop_path=dp_rates[23 + j],
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[3]) for j in range(depths[3])],
            nn.Conv3d(c_list[4], c_list[5], 3, stride=1, padding=1),
        )

        if bridge:
            self.scab = SC_Att_Bridge3D(c_list, split_att)

        self.decoder1 = nn.Sequential(
            nn.Conv3d(c_list[5], c_list[4], 3, stride=1, padding=1),
            *[block(dim=c_list[4], drop_path=dp_rates[24] if len(dp_rates) > 24 else 0.0,
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[3]) for _ in range(depths[3])],
        )

        self.decoder2 = nn.Sequential(
            nn.Conv3d(c_list[4], c_list[3], 3, stride=1, padding=1),
            *[block(dim=c_list[3], drop_path=dp_rates[5 + j] if (5 + j) < len(dp_rates) else 0.0,
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[2]) for j in range(depths[2])],
        )

        self.decoder3 = nn.Sequential(
            nn.Conv3d(c_list[3], c_list[2], 3, stride=1, padding=1),
            *[block(dim=c_list[2], drop_path=dp_rates[2 + j] if (2 + j) < len(dp_rates) else 0.0,
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[1]) for j in range(depths[1])],
        )

        self.decoder4 = nn.Sequential(
            nn.Conv3d(c_list[2], c_list[1], 3, stride=1, padding=1),
            *[block(dim=c_list[1], drop_path=dp_rates[0 + j] if (0 + j) < len(dp_rates) else 0.0,
                    layer_scale_init_value=layer_scale_init_value, gnconv=gnconvs[0]) for j in range(depths[0])],
        )

        self.decoder5 = nn.Sequential(
            nn.Conv3d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        # GroupNorm works for 3D too
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv3d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            k = m.kernel_size
            if isinstance(k, tuple):
                fan_out = 1
                for kk in k:
                    fan_out *= kk
                fan_out *= m.out_channels
            else:
                fan_out = k * k * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _upsample(self, x, scale_factor):
        return F.interpolate(
            x, scale_factor=scale_factor,
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ["trilinear"] else None
        )

    def forward(self, x):
        # accept [B,C,H,W,D] like your UNet3D demo
        if x.ndim != 5:
            raise ValueError(f"Expect 5D input [B,C,H,W,D], got {x.shape}")

        # [B,C,H,W,D] -> [B,C,D,H,W]
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        # encoder (downsample 5 times -> /32 on D/H/W)
        out = F.gelu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # [B,c0,D/2,H/2,W/2]

        out = F.gelu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # [B,c1,D/4,H/4,W/4]

        out = F.gelu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # [B,c2,D/8,H/8,W/8]

        out = F.gelu(F.max_pool3d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # [B,c3,D/16,H/16,W/16]

        out = F.gelu(F.max_pool3d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # [B,c4,D/32,H/32,W/32]

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # [B,c5,D/32,H/32,W/32]

        # decoder
        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # [B,c4,D/32,H/32,W/32]
        out5 = out5 + t5

        out4 = F.gelu(self._upsample(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2, 2)))
        out4 = out4 + t4

        out3 = F.gelu(self._upsample(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2, 2)))
        out3 = out3 + t3

        out2 = F.gelu(self._upsample(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2, 2)))
        out2 = out2 + t2

        out1 = F.gelu(self._upsample(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2, 2)))
        out1 = out1 + t1

        out0 = self._upsample(self.final(out1), scale_factor=(2, 2, 2))  # back to (D,H,W)

        # return logits [B,num_classes,D,H,W] (UNet3D-style)
        if self.apply_sigmoid:
            out0 = torch.sigmoid(out0)
        return out0


if __name__ == "__main__":
    # Like UNet3D demo: input [B,C,H,W,D]
    B, C, H, W, D = 2, 1, 128, 128, 128
    x = torch.randn(B, C, H, W, D).cuda()

    net = MHorunet3D(
        num_classes=2,
        input_channels=C,
        c_list=[8, 16, 32, 64, 128, 256],
        depths=[2, 3, 18, 2],
        drop_path_rate=0.0,
        bridge=True,
        apply_sigmoid=False,   # training建议 logits
        upsample_mode="trilinear",
    ).cuda()

    y = net(x)
    print("output:", y.shape)  # [B,2,D,H,W]
