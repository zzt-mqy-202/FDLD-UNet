"""3D VoxelMorph model.

Inputs are 3D tensors [B, C, D, H, W]. The model returns the warped moving
image and a 3-channel displacement field [B, 3, D, H, W].
"""
try:
    from ..model2d.voxelmorph import VoxelMorphUNet as _VoxelMorphUNet
    from ..model2d.voxelmorph import VoxelMorph as _VoxelMorph
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model2d"))
    from voxelmorph import VoxelMorphUNet as _VoxelMorphUNet
    from voxelmorph import VoxelMorph as _VoxelMorph

__all__ = ["VoxelMorphUNet", "VoxelMorph", "voxelmorphunet", "voxelmorph"]


class VoxelMorphUNet(_VoxelMorphUNet):
    """3D VoxelMorph UNet backbone."""

    def __init__(
        self,
        in_channels=2,
        unet_out_channels=32,
        channels=(16, 32, 32, 32, 32, 32),
        final_conv_channels=(16, 16),
        **kwargs,
    ):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            unet_out_channels=unet_out_channels,
            channels=channels,
            final_conv_channels=final_conv_channels,
            **kwargs,
        )


class VoxelMorph(_VoxelMorph):
    """3D VoxelMorph registration network."""

    def __init__(self, backbone=None, integration_steps=7, half_res=False):
        if backbone is None:
            backbone = VoxelMorphUNet()
        super().__init__(
            backbone=backbone,
            integration_steps=integration_steps,
            half_res=half_res,
            spatial_dims=3,
        )


voxelmorphunet = VoxelMorphUNet
voxelmorph = VoxelMorph


if __name__ == "__main__":
    import torch

    net = VoxelMorph(
        backbone=VoxelMorphUNet(in_channels=2),
        integration_steps=7,
        half_res=False,
    )
    moving = torch.randn(1, 1, 64, 128, 128)
    fixed = torch.randn(1, 1, 64, 128, 128)
    warped, ddf = net(moving, fixed)
    print("warped:", warped.shape, "ddf:", ddf.shape)
