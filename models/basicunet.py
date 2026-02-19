from monai.networks.nets import BasicUNet
from timm.models import register_model

__all__ = ['MONAI_BasicUNet',]


def _cfg():
    return dict(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        features=(32, 32, 64, 128, 256, 32),
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        bias=True,
        dropout=0.0,
        upsample="deconv",
    )


class MONAI_BasicUNet(BasicUNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            spatial_dims=kwargs['spatial_dims'],
            in_channels=kwargs['in_channels'],
            out_channels=kwargs['num_classes'],
            features=kwargs['features'],
            act=kwargs['act'],
            norm=kwargs['norm'],
            bias=kwargs['bias'],
            dropout=kwargs['dropout'],
            upsample=kwargs['upsample'],
        )


@register_model
def monai_basicunet(**kwargs):
    cfg = _cfg()
    cfg.update(**kwargs)
    model = MONAI_BasicUNet(**cfg)
    return model
