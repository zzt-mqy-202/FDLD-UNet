
from timm import create_model

from .D2_FDLD import FDLD_UNet
from .D3_FDLD import FDLD


def build_model(name, spatial_dims=2, in_channels=3, num_classes=2, img_size=256, **kwargs):
    """Build a main model using the project's dimensionality convention."""
    key = str(name).lower()
    if key in {"fdld_unet", "d2_fdld", "fdld2d"}:
        if spatial_dims != 2:
            raise ValueError("D2_FDLD requires spatial_dims=2")
        return FDLD_UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            **kwargs,
        )
    if key in {"d3_fdld", "fdld3d", "fdld"}:
        if spatial_dims != 3:
            raise ValueError("D3_FDLD requires spatial_dims=3")
        return FDLD(
            n_channels=in_channels,
            n_classes=num_classes,
            **kwargs,
        )
    return create_model(
        name,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )


__all__ = ["create_model", "build_model", "FDLD_UNet", "FDLD"]
