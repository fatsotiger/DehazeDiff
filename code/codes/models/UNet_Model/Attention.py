import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange



class CgaAttenaiton(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CgaAttenaiton, self).__init__()
        # self.conv1 = nn.Conv2d(dim, dim, 3,1,1, bias=True)
        # self.act1 = nn.ReLU(inplace=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, initial,act = True):
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        w = self.sigmoid(self.pa(initial, pattn1))
        if act : return initial*w
        # result = initial + pattn2 * x + (1 - pattn2) * y
        # result = self.conv(result)
        else: return w

class CgaAttenaiton_updata(nn.Module):
    def __init__(self, dim, reduction=2):

        super(CgaAttenaiton_updata, self).__init__()
        # self.conv1 = nn.Conv2d(dim, dim, 3,1,1, bias=True)
        # self.act1 = nn.ReLU(inplace=True)
        self.dim = dim

        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim//2, reduction)
        self.pa = PixelAttention(dim)

        self.sigmoid = nn.Sigmoid()

        self.sigmoid_1 = nn.Sigmoid()
        self.conv_1 = nn.Conv2d(dim//2,dim//2,1,1,0)
        self.sigmoid_2 = nn.Sigmoid()
        self.conv_2 = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim,dim//2,1,1,0),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim,dim//2,1,1,0),
            nn.LeakyReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(dim, dim//2, 7, 1, 3),
            # nn.BatchNorm2d(dim//2),
            nn.Tanh() ,

        )
        self.pa = PixelAttention(dim)

    def forward(self, initial):
        # print(initial.shape)
        x1,x2 = torch.split(initial,self.dim//2,dim=1)
        # print(x1.shape)
        # precess x1
        a = self.sigmoid_1(self.conv_1(x1))
        x1 =x1*self.ca(x1)


        # precess x2
        b = self.sigmoid_1(self.conv_1(x2))
        x2 = x2*self.sa(x2)


        # print(a.shape,b.shape)


        x1 = self.conv1(torch.cat((a*x1 ,(1-b)*x2),dim=1))
        x2 = self.conv2(torch.cat(((1-a)*x1 , b*x2),dim=1))

        output = self.output(torch.cat((x1,x2),dim=1)+ initial)


        return output

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        # print(x.shape)
        x_gap = self.gap(x)
        # print(x_gap.shape)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        # pattn2 = self.sigmoid(pattn2)
        return pattn2


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

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
class CP_Attention_block(nn.Module):
    def __init__(self, dim=64, kernel_size=3):
        super(CP_Attention_block, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,padding=1)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res