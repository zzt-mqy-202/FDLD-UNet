import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.1),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        if out.size()[-1] != 126:
            return torch.cat((out,feature_map),dim=1)
        else:
            return torch.cat((out[:,:,:-1,:-1], feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self,inchannel,n_classes):
        super(UNet, self).__init__()
        self.inchannel = inchannel
        self.n_classes = n_classes
        self.c1=Conv_Block(self.inchannel,32)
        self.d1=DownSample(32)
        self.c2=Conv_Block(32,64)
        self.d2=DownSample(64)
        self.c3=Conv_Block(64,128)
        self.d3=DownSample(128)
        self.c4=Conv_Block(128,256)
        self.d4=DownSample(256)
        self.c5=Conv_Block(256,512)

        self.u1=UpSample(512)
        self.c6=Conv_Block(512,256)
        self.u2 = UpSample(256)
        self.c7 = Conv_Block(256, 128)
        self.u3 = UpSample(128)
        self.c8 = Conv_Block(128, 64)
        self.u4 = UpSample(64)
        self.c9 = Conv_Block(64, 32)
        self.out=nn.Conv2d(32,self.n_classes,3,1,1)
        self.Th=nn.Sigmoid()

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))

if __name__ == '__main__':
    data = torch.randn(1, 3, 256, 256).cuda()


    net=UNet(inchannel=3,n_classes = 2).cuda()

    # print(sum(p.numel() for p in net.parameters()))
    print(net(data).shape)

    from thop import profile
    flops, params = profile(net, (data,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')