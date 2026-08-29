import torch
import torch.nn as nn


from D2_voxelmorph import VoxelMorph

from D3_vmunet import VMUNet3D as VMamba3D


class VMambaFlowBackbone3D(nn.Module):
    """VMamba 3D wrapper that predicts a 3-channel 3D displacement field."""

    def __init__(
        self,
        in_channels=2,
        flow_channels=3,
        depths=(1, 2, 6),
        depths_decoder=(6, 2, 1),
        drop_path_rate=0.2,
        pad_multiple_hw=8,
    ):
        super().__init__()
        if flow_channels != 3:
            raise ValueError("3D displacement fields must have 3 channels")
        self.vmamba = VMamba3D(
            input_channels=in_channels,
            num_classes=flow_channels,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            drop_path_rate=drop_path_rate,
            pad_multiple_hw=pad_multiple_hw,
            apply_sigmoid=False,
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
        # vmunet_D3 returns [B,C,H,W,D]; VoxelMorph expects [B,C,D,H,W].
        y = self.vmamba(x)
        return y.permute(0, 1, 4, 2, 3).contiguous()


class VMambaMorphVoxelMorph3D(nn.Module):
    """3D VMambaMorph registration network."""

    def __init__(
        self,
        image_channels=1,
        depths=(1, 2, 6),
        depths_decoder=(6, 2, 1),
        drop_path_rate=0.2,
        pad_multiple_hw=8,
        integration_steps=7,
        half_res=False,
    ):
        super().__init__()
        self.backbone = VMambaFlowBackbone3D(
            in_channels=image_channels * 2,
            flow_channels=3,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
            pad_multiple_hw=pad_multiple_hw,
        )
        self.registration = VoxelMorph(
            backbone=self.backbone,
            integration_steps=integration_steps,
            half_res=half_res,
        )

    def forward(self, moving, fixed):
        if moving.ndim != 5 or fixed.ndim != 5:
            raise ValueError("moving and fixed must be [B,C,D,H,W]")
        return self.registration(moving, fixed)


VMambaMorph3D = VMambaMorphVoxelMorph3D


if __name__ == "__main__":
    moving = torch.randn(1, 1, 16, 128, 128)
    fixed = torch.randn(1, 1, 16, 128, 128)
    model = VMambaMorphVoxelMorph3D(image_channels=1)
    warped, ddf = model(moving, fixed)
    print("warped:", warped.shape, "ddf:", ddf.shape)
