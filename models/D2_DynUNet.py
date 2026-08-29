import torch.nn as nn
from monai.networks.nets import DynUNet
class DynUNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,**kwargs):
        super().__init__(); self.net=DynUNet(spatial_dims=2,in_channels=in_channels,out_channels=out_channels,kernel_size=[3]*5,strides=[1,2,2,2,2],upsample_kernel_size=[2]*5)
    def forward(self,x): return self.net(x)
