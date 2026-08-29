import torch.nn as nn
from monai.networks.nets import FlexUNet
class FlexUNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,**kwargs):
        super().__init__(); self.net=FlexUNet(spatial_dims=2,in_channels=in_channels,out_channels=out_channels,pretrained=False,backbone='resnet10')
    def forward(self,x): return self.net(x)
