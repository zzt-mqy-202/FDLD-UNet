"""2D VoxelMorph model.

Inputs are 2D tensors [B, C, H, W]. The model returns the warped moving
image and a 2-channel displacement field [B, 2, H, W].
"""
try:
    from .voxelmorph import VoxelMorphUNet as _VoxelMorphUNet
    from .voxelmorph import VoxelMorph as _VoxelMorph
except ImportError:
    from voxelmorph import VoxelMorphUNet as _VoxelMorphUNet
    from voxelmorph import VoxelMorph as _VoxelMorph

import torch
import torch.nn.functional as F

__all__ = ["VoxelMorphUNet", "VoxelMorph", "voxelmorphunet", "voxelmorph"]


class VoxelMorphUNet(_VoxelMorphUNet):
    """2D VoxelMorph UNet backbone.

    ``in_channels`` is the number of channels after concatenating moving and
    fixed images; for two single-channel images it should be 2.
    """

    def __init__(
        self,
        in_channels=2,
        unet_out_channels=32,
        channels=(16, 32, 32, 32, 32, 32),
        final_conv_channels=(16, 16),
        **kwargs,
    ):
        super().__init__(
            spatial_dims=2,
            in_channels=in_channels,
            unet_out_channels=unet_out_channels,
            channels=channels,
            final_conv_channels=final_conv_channels,
            **kwargs,
        )


class VoxelMorph(_VoxelMorph):
    """2D VoxelMorph registration network."""

    def __init__(self, backbone=None, integration_steps=7, half_res=False):
        if backbone is None:
            backbone = VoxelMorphUNet()
        super().__init__(
            backbone=backbone,
            integration_steps=integration_steps,
            half_res=half_res,
            spatial_dims=2,
        )

    def forward(self, moving, fixed):
        """Run registration with 2D interpolation when ``half_res`` is used."""
        if not self.half_res:
            return super().forward(moving, fixed)
        if moving.shape != fixed.shape:
            raise ValueError("moving and fixed images must have the same shape")
        x = self.backbone(torch.cat([moving, fixed], dim=1))
        if x.shape[1] != 2 or x.shape[2:] != moving.shape[2:]:
            raise ValueError("backbone must return [B, 2, H, W] at input resolution")
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True) * 2.0
        if self.diffeomorphic:
            x = self.dvf2ddf(x)
        x = F.interpolate(x * 0.5, scale_factor=2.0, mode="bilinear", align_corners=True)
        return self.warp(moving, x), x


voxelmorphunet = VoxelMorphUNet
voxelmorph = VoxelMorph


if __name__ == "__main__":
    import torch

    net = VoxelMorph(
        backbone=VoxelMorphUNet(in_channels=2),
        integration_steps=7,
        half_res=False,
    )
    moving = torch.randn(1, 1, 128, 128)
    fixed = torch.randn(1, 1, 128, 128)
    warped, ddf = net(moving, fixed)
    print("warped:", warped.shape, "ddf:", ddf.shape)
