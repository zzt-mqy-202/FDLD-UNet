import torch.nn as nn
from monai.networks.nets import SwinUNETR
class SwinUNETR2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,feature_size=12,**kwargs):
        super().__init__(); self.net=SwinUNETR(in_channels=in_channels,out_channels=out_channels,spatial_dims=2,feature_size=feature_size,**{k:v for k,v in kwargs.items() if k in {"drop_rate","attn_drop_rate","dropout_path_rate","use_checkpoint"}})
    def forward(self,x): return self.net(x)
