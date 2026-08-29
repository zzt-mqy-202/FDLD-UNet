import torch
import torch.nn as nn

from D2_voxelmorph import VoxelMorph

from D3_vmunet import VMUNet as VMamba2D



class VMambaFlowBackbone2D(nn.Module):
    """VMamba backbone that predicts a 2-channel 2D displacement field."""

    def __init__(
        self,
        in_channels=2,
        flow_channels=2,
        depths=(2, 2, 9, 2),
        depths_decoder=(2, 9, 2, 2),
        drop_path_rate=0.2,
    ):
        super().__init__()
        if flow_channels != 2:
            raise ValueError("2D displacement fields must have 2 channels")
        self.vmamba = VMamba2D(
            input_channels=in_channels,
            num_classes=flow_channels,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            drop_path_rate=drop_path_rate,
            apply_sigmoid=False,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        return self.vmamba(x)


class VMambaMorphVoxelMorph2D(nn.Module):
    """2D VMambaMorph registration network."""

    def __init__(
        self,
        image_channels=1,
        depths=(2, 2, 9, 2),
        depths_decoder=(2, 9, 2, 2),
        drop_path_rate=0.2,
        integration_steps=7,
        half_res=False,
    ):
        super().__init__()
        self.backbone = VMambaFlowBackbone2D(
            in_channels=image_channels * 2,
            flow_channels=2,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )
        self.registration = VoxelMorph(
            backbone=self.backbone,
            integration_steps=integration_steps,
            half_res=half_res,
        )

    def forward(self, moving, fixed):
        if moving.ndim != 4 or fixed.ndim != 4:
            raise ValueError("moving and fixed must be [B,C,H,W]")
        return self.registration(moving, fixed)


VMambaMorph2D = VMambaMorphVoxelMorph2D


if __name__ == "__main__":
    moving = torch.randn(1, 1, 128, 128)
    fixed = torch.randn(1, 1, 128, 128)
    model = VMambaMorphVoxelMorph2D(image_channels=1)
    warped, ddf = model(moving, fixed)
    print("warped:", warped.shape, "ddf:", ddf.shape)
