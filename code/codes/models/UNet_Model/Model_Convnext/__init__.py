import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from torchsummary import summary


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm,self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super( Block,self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # print(x.shape)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class proBlock(nn.Module):
    def __init__(self,inputchannal):
        super(proBlock,self).__init__()
        self.convdown4b = nn.Conv2d(in_channels=inputchannal,out_channels=128
                                    ,kernel_size=4,stride=4,padding=0)
        self.convdown2d = nn.Conv2d(in_channels=inputchannal,out_channels=32,kernel_size=2,stride=2,padding=0)
        self.CA = CALayer(64)
        self.conv = nn.Conv2d(in_channels=inputchannal,out_channels=16,kernel_size=3,stride=1,padding=1)

        self.up = nn.PixelShuffle(2)


    def forward(self, x):
        # print(x.shape)
        x4d = self.convdown4b(x)

        x4d = self.up(x4d)
        x2d = self.convdown2d(x)
        x1 = self.CA(torch.cat((x4d,x2d),1))
        output = torch.cat((self.up(x1),self.conv(x)),1)
        # print(output.shape)

        return output
class physicalpriorBlock(nn.Module):
    def __init__(self,inchannel = 3):
        super(physicalpriorBlock,self).__init__()
        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=inchannel,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),



        )
        self.conv2= nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),


        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()

        )
        # self.k = nn.Sequential(
        #
        #     nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,stride=1,padding=1),
        #     nn.Sigmoid()
        # )



    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1
        k = self.conv3(x2)
        output = k * x - k + 1

        return output
class Bottle2neck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(inplanes/4)
        # print('width',width)
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
                # print(sp.shape)
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.conv3(out)
        out = self.bn3(out)


        out = self.relu(out)
        return out
class Convnextpro(nn.Module):
    def __init__(self,inputchannal):
        super(Convnextpro,self).__init__()
        self.Convinput = proBlock(inputchannal)

        self.Conv1 = Block(32)
        self.Conv2 = Block(32)
        self.Conv3 = Block(32)

        self.dropout = nn.Dropout2d(p=0.2)

        # self.dropout = nn.Dropout2d(p=0.2)

        self.ConvLayer2 = nn.Sequential(
            Block(96),
            nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=0),
            Block(192),
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0),
            Block(96),
        )

        self.detailblock = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=7,stride=1,padding=3),
            Bottle2neck(64,64),
            Bottle2neck(64, 64),
            Bottle2neck(64, 64),
        )
        self.Convoutput = nn.Sequential(
            nn.Conv2d(96+64, 96, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(96, 125, kernel_size=3, stride=1, padding=1),

        )
        # self.phyblock = physicalpriorBlock(3)


    def forward(self, x):
        x1 = self.Convinput(x)

        x2 = self.Conv1(x1)
        x3 = self.Conv2(x2)
        x4 = self.Conv3(x3)

        x4 = self.dropout(torch.cat((x2,x3,x4),1))
        # print(x4.shape)

        x5 = self.ConvLayer2(x4)
        xd = self.detailblock(x1)
        # print(xd.shape)
        output = self.Convoutput(torch.cat((x5,xd),1))
        output = torch.cat((output,x),1)
        # print(output.shape)

        # output = self.phyblock(output) + output

        return output

if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net2 = Convnextpro().to(device)
    summary(net2, input_size=(3, 256, 192), batch_size=1)
