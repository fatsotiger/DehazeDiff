import torch
import torch.nn as nn
from torchsummary import summary

from models.UNet_Model.Attention import CgaAttenaiton, CP_Attention_block, CALayer
from models.UNet_Model.GRN_Model import Block_updata,LayerNorm,GRN
from models.UNet_Model.Model_RES2NET.CodeBook import knowledge_CodeBook
from models.UNet_Model.PriorFusion import PriorFusion
from models.UNet_Model.COnvnext import knowledge_adaptation_convnext
from models.UNet_Model.Decoder_ import ResBlock


class Base(nn.Module):
    def __init__(self):
        super(Base,self).__init__()
        pass

    def forward(self,input):
        pass
        return 0

# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
#
#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x
import torch.nn.functional as F

class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

import  seaborn as sns
import numpy as np
import matplotlib .pylab as plt
from mpl_toolkits.mplot3d import Axes3D
class Feture_process(nn.Module):
    """
        特征压缩：两个子块，1>通道信息纠缠块 2>特征压缩块

    """
    def __init__(self,inchannel = 512):
        super(Feture_process,self).__init__()
        # 特征压缩
        self.conv1_2 = nn.Conv2d(inchannel+8,inchannel // 2,3,1,1) # 512->256
        self.conv1_4 = nn.Conv2d(inchannel+8, inchannel // 4, 3, 1, 1) # 521->128
        self.conv1_8 = nn.Conv2d(inchannel+8, inchannel // 8, 3, 1, 1) # 512->32

        self.conv2_2 = nn.Conv2d(inchannel // 2 , inchannel // 4, 3, 1, 1) # 256->128

        self.conv3_4 = nn.Conv2d(inchannel // 2 , inchannel // 8, 3, 1, 1) # 128+128->64

        self.conv4_4 = nn.Conv2d(inchannel // 8, inchannel // 32, 3, 1, 1)  # 64->16

        self.ca = nn.Sequential(
            nn.Conv2d(inchannel//4, inchannel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // 8, inchannel //8 , 1, padding=0, bias=True),
            nn.Sigmoid()
        ) # 64+64 -> 64

        # 通道信息纠缠
        self.f1 = Feture_min(k = 8) # 512/64 = 8
        # self.f2 = Feture_min(k=2)
        # self.conv_f = nn.Conv2d(inchannel//16 + inchannel//64,inchannel//32,3,1,1)


    def forward(self,x):
        # 通道信息纠缠
        f1 = self.f1(x)

        #特征压缩
        x0 = torch.cat((f1,x),1)
        x_1 = self.conv1_2(x0)
        x_2 = self.conv1_4(x0)
        x_3 = self.conv1_8(x0)

        x_12 = torch.cat((x_2,self.conv2_2(x_1)),1)
        x_12 = self.conv3_4(x_12)

        w = self.ca(torch.cat((x_12,x_3),1))
        x_ = x_3*w + x_12*(1-w)
        x_ = self.conv4_4(x_)


        # f2 = self.f2(f1)
        # x__ = self.conv_f(torch.cat((f1,f2),1))


        # x = torch.cat((x_,f1),1) # 16 + 8 = 24

        return x_,f1

class Feture_deprocess(nn.Module):
    """
        特征解压：两个子块，1>解压通道信息纠缠块 2>解压特征压缩块

    """

    def __init__(self, inchannel=16):
        super(Feture_deprocess, self).__init__()
        # 特征压缩
        self.conv1_2 = nn.Conv2d(inchannel + 8, inchannel * 2, 3, 1, 1)  # 16->32
        self.conv1_4 = nn.Conv2d(inchannel + 8, inchannel * 4, 3, 1, 1)  # 16->64
        self.conv1_8 = nn.Conv2d(inchannel + 8, inchannel * 8, 3, 1, 1)  # 16->128

        self.conv2_2 = nn.Conv2d(inchannel * 2, inchannel * 4, 3, 1, 1)  # 32->64

        self.conv3_4 = nn.Conv2d(inchannel * 8, inchannel * 8, 3, 1, 1)  # 128->128

        self.conv4_4 = nn.Conv2d(inchannel * 16, inchannel *32, 3, 1, 1)  # 128+128->512

        self.ca = nn.Sequential(
            nn.Conv2d(inchannel * 16, inchannel * 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel * 8, inchannel * 8, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Conv2d(inchannel * 64,inchannel *32,1,1,0),
            nn.PReLU(),
            nn.Conv2d(inchannel * 32, inchannel * 64, 1, 1, 0),
        )
        # 通道信息纠缠
        self.f1 = Feture_magnify(k=8)  # 512/64 = 8
        # self.f2 = Feture_min(k=2)
        # self.conv_f = nn.Conv2d(inchannel//16 + inchannel//64,inchannel//32,3,1,1)

    def forward(self, x,c):
        # 通道信息纠缠
        f1 = self.f1(c)
        # print(f1.shape)

        # 特征压缩
        x0 = torch.cat((c, x), 1)
        x_1 = self.conv1_2(x0)
        x_2 = self.conv1_4(x0) # 64
        x_3 = self.conv1_8(x0)

        x_12 = torch.cat((x_2, self.conv2_2(x_1)), 1)
        x_12 = self.conv3_4(x_12)
        # print(x_12.shape,x_3.shape)

        x_ = torch.cat((x_12, x_3), 1)
       #  x_ = x_3 * w + x_12 * (1 - w)
        # print(x_.shape)
        x_ = self.conv4_4(x_)
        # print(x_.shape)

        x = torch.cat((x_, f1), 1)  # 1024
        x = self.output(x)

        return x
# class Feture_deprocess(nn.Module):
#     """
#         特征解压：两个子块，1>解压通道信息纠缠块 2>解压特征压缩块
#
#     """
#
#     def __init__(self, inchannel=16):
#         super(Feture_deprocess, self).__init__()
#         # 特征压缩
#         self.conv1_2 = nn.Conv2d(inchannel + 8, inchannel * 2, 3, 1, 1)  # 16->32
#         self.conv1_4 = nn.Conv2d(inchannel + 8, inchannel * 4, 3, 1, 1)  # 16->64
#         self.conv1_8 = nn.Conv2d(inchannel + 8, inchannel * 8, 3, 1, 1)  # 16->128
#
#         self.conv2_2 = nn.Conv2d(inchannel * 2, inchannel * 4, 3, 1, 1)  # 32->64
#
#         self.conv3_4 = nn.Conv2d(inchannel * 8, inchannel * 8, 3, 1, 1)  # 128->128
#
#         self.conv4_4 = nn.Conv2d(inchannel * 8, inchannel *32, 3, 1, 1)  # 128+128->512
#
#         self.ca = nn.Sequential(
#             nn.Conv2d(inchannel * 16, inchannel * 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inchannel * 8, inchannel * 8, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#         self.output = nn.Sequential(
#             nn.Conv2d(inchannel * 64,inchannel *32,1,1,0),
#             nn.PReLU(),
#             nn.Conv2d(inchannel * 32, inchannel * 64, 1, 1, 0),
#         )
#         # 通道信息纠缠
#         self.f1 = Feture_magnify(k=8)  # 512/64 = 8
#         # self.f2 = Feture_min(k=2)
#         # self.conv_f = nn.Conv2d(inchannel//16 + inchannel//64,inchannel//32,3,1,1)
#
#     def forward(self, x,c):
#         # 通道信息纠缠
#         f1 = self.f1(c)
#         # print(f1.shape)
#
#         # 特征压缩
#         x0 = torch.cat((c, x), 1)
#         x_1 = self.conv1_2(x0)
#         x_2 = self.conv1_4(x0) # 64
#         x_3 = self.conv1_8(x0)
#
#         x_12 = torch.cat((x_2, self.conv2_2(x_1)), 1)
#         x_12 = self.conv3_4(x_12)
#         # print(x_12.shape,x_3.shape)
#
#         w = self.ca(torch.cat((x_12, x_3), 1))
#         x_ = x_3 * w + x_12 * (1 - w)
#         # print(x_.shape)
#         x_ = self.conv4_4(x_)
#         # print(x_.shape)
#
#         x = torch.cat((x_, f1), 1)  # 1024
#         x = self.output(x)
#
#         return x
class Encode(nn.Module):
    def __init__(self):
        super(Encode,self).__init__()

        self.prior = nn.Sequential(
            # knowledge_CodeBook(),
            knowledge_adaptation_convnext(),
        )
        # self.knowledge_adaptation_branch = knowledge_adaptation_convnext()


        # self.convx2 = nn.Conv2d(256*2,256,3,1,1)
        self.RDB = nn.Sequential(
            RDB(512,3,32),
            CP_Attention_block(512),

        )
        self.fsu = PriorFusion(16)
        self.minM1 = Feture_min(k = 4)
        self.minM2 = Feture_min(k=2)
        self.channalshuffle = ChannelShuffle(4)

        self.f = Feture_process(512)

    def forward(self,input):
        # x1 = self.input(input) # 1/2
        # x2 = self.conv1(x1) # 1/4
        # out = self.conv2(x2) # 1/8



        x2,x3 = self.prior(input) # torch.Size([2, 64, 256, 256]) torch.Size([2, 256, 128, 128]) torch.Size([2, 512, 64, 64])
        # print(x1.shape,x2.shape,x3.shape)

        # x3_ = self.RDB(x3)
        x3,channalpre = self.f(x3) # [1, 16, 96, 96],8
        x1 = x3
        # prio = self.PriorDown(torch.cat((convx,prio),dim=1))
        # print(prio.shape)

        # print(x3.shape)
        # t = self.minM1(x3)
        # print(t.shape)
        # prio = self.minM2(t) # c = 16
        # x1 = prio
        # print(prio.shape)
        # prio = self.fsu(prio)

         # 1,512,64,64

        # Fusion_data = self.fsu(prio, prio)
        # Fusion_data = out
        # print('out = ',out.shape)


        return  x1,x2,x3,channalpre

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        # transpose
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)

        return x
class Feture_magnify(nn.Module):
    """
        特征放大器

    """
    def __init__(self,k):
        super(Feture_magnify,self).__init__()
        # self.conv1 = nn.Conv2d(256,512,3,1,1)


        self.up = nn.ConvTranspose2d(1,1,k,k)
        self.channalshuffle = ChannelShuffle(4)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 1, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self,x):
        # y = torch.cat((x,x),dim=1)
        #  y = self.channalshuffle(y)
        b,c,w,h = x.shape
        x1 = x.view(b,1,c,w*h)
        x1 = self.up(x1)
        weight = self.ca(x1)
        x1 = x1 * weight
        # print(x1.shape)
        x1 = x1.view(b, -1, w, h)
        # print(x1.shape)
        # out = self.conv1(x1)



        return x1
# x = torch.randn(1,8,64,64)
# f = Feture_magnify()
# print(f(x).shape)
class Feture_min(nn.Module):
    """
        特征缩小器

    """
    def __init__(self,k=2,dim = 1):
        super(Feture_min,self).__init__()
        self.conv1 = nn.Conv2d(1,1,k,k)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.ca =  nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 1, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,w,h = x.shape
        x = x.view(b,1,c,w*h)
        x0 = self.conv1(x) # down 4倍
        weight = self.ca(x0)
        x0 = x0*weight

        x = x0.permute(0,2,3,1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = x + x0

        x = x.view(b, -1, w, h)

        return  x
def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size//2), bias=bias)
class Decode(nn.Module):
    def __init__(self):
        super(Decode,self).__init__()
        self.pconv1 =nn.Conv2d(16,64,1,1,0)

        self.f =  Feture_deprocess(16)

        self.upconv1 = nn.Sequential(
            # nn.ConvTranspose2d(512,64,kernel_size=2,stride=2),
            #  CP_Attention_block(5),
            # nn.Conv2d(512, 1024, 1, 1, 0),

            nn.PixelShuffle(2),
            CP_Attention_block(256),

            # Block_updata(256),
            # nn.Conv2d(128, 256, 3, 1, 1),
            # nn.PReLU(),

        )
        self.upconv2 = nn.Sequential(
            #  nn.ConvTranspose2d(32*2, 32*2, kernel_size=2, stride=2),
            # CP_Attention_block(32*2),
            #nn.Conv2d(32*2, 16, 3, 1, 1),
            # nn.Conv2d(256*2, 256, 3, 1, 1),
            ResBlock(conv=default_conv,dim_in=256*2, dim_out=256),
            nn.PixelShuffle(2),
            ResBlock(conv=default_conv, dim_in=64, dim_out=64),
            CP_Attention_block(64),
            # Block_updata(64),
            # nn.Conv2d(32, 64, 3, 1, 1),
            # nn.PReLU(),

        )
        self.upconv3 = nn.Sequential(
            # nn.ConvTranspose2d(16*2, 16*2, kernel_size=2, stride=2),
            # CP_Attention_block(16*2),
            # nn.Conv2d(64, 64 ,3, 1, 1),
            ResBlock(conv=default_conv, dim_in=64, dim_out=64),
            nn.PixelShuffle(2),

            ResBlock(conv=default_conv, dim_in=16, dim_out=3),
            nn.Tanh(),

        )


    def forward(self,input,encode_x1,encode_x2,encode_x3,channalpre):
        # f1 = self.pconv1(Fusion_data)
        # f2 = self.pconv2(f1)
        # f3 = self.pconv3(f2)
        # Fusion_data = torch.cat((Fusion_data,f1,f2,f3),dim=1)
        # print(Fusion_data.shape)
        # print(Fusion_data.shape)

        # plt.subplot(1, 2, 1)
        # x_3 = Fusion_data
        #
        # # print(x_3[0].squeeze(0).shape)
        # data = x_3[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # data = np.mean(data,axis=2)
        # # # 创建图形和3D轴
        # # data = x3
        # plt.imshow(data, cmap='jet', interpolation='nearest')



        # f1 = self.mag1(Fusion_data)
        # Fusion_data = self.mag2(f1) # c = 64
        #
        #
        # x3 = torch.cat((Fusion_data,encode_x3),dim=1)
        x3 = self.f(encode_x3,channalpre)
        x1 = self.upconv1(x3)

        x2 = self.upconv2(torch.cat((x1,encode_x2),dim=1))
        # out = self.upconv3(torch.cat((x2,encode_x1),dim=1))
        out = self.upconv3(x2)
        # out = input + out

        # plt.subplot(1, 2, 2)
        # p = encode_x3
        # data = p[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # data = np.mean(data, axis=2)
        # # # 创建图形和3D轴
        # # data = x3
        # plt.imshow(data, cmap='jet', interpolation='nearest')
        # plt.show()


        # out = input + out

        return  out
# class Encode(nn.Module):
#     def __init__(self):
#         super(Encode,self).__init__()
#         self.input = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 2, 1),
#             # nn.Conv2d(64, 128, 3, 2, 1),
#             Block_updata(64),
#
#             # nn.PReLU(),
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 2, 1),
#             # nn.Conv2d(256, 512, 3, 2, 1),
#             Block_updata(128),
#
#             # nn.PReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, 2, 1),
#             Block_updata(256),
#
#             # nn.PReLU(),
#         )
#
#         self.prior = nn.Sequential(
#             knowledge_CodeBook(),
#             nn.Conv2d(512, 256, 3, 1, 1),
#             # nn.Conv2d(256, 64, 3, 1, 1),
#             # nn.Conv2d(64, 8, 3, 1, 1),
#         )
#         self.fsu = PriorFusion(256)
#
#     def forward(self,input):
#         x1 = self.input(input) # 1/2
#         x2 = self.conv1(x1) # 1/4
#         out = self.conv2(x2) # 1/8
#
#         prio = self.prior(input)
#         Fusion_data = self.fsu(out, prio)
#         # Fusion_data = out
#         # print('out = ',out.shape)
#         return  x1,x2,Fusion_data,Fusion_data
#
# class Decode(nn.Module):
#     def __init__(self):
#         super(Decode,self).__init__()
#         self.upconv1 = nn.Sequential(
#             nn.ConvTranspose2d(256*2,128,kernel_size=2,stride=2),
#             # nn.PixelShuffle(2),
#             CP_Attention_block(128),
#             Block_updata(128),
#             # nn.Conv2d(256, 256, 3, 1, 1),
#             # nn.PReLU(),
#
#         )
#         self.upconv2 = nn.Sequential(
#             nn.ConvTranspose2d(128*2, 64, kernel_size=2, stride=2),
#             # nn.PixelShuffle(2),
#             CP_Attention_block(64),
#             Block_updata(64),
#             # nn.Conv2d(64, 64, 3, 1, 1),
#             # nn.PReLU(),
#
#         )
#         self.upconv3 = nn.Sequential(
#             nn.ConvTranspose2d(64*2, 16, kernel_size=2, stride=2),
#             #nn.PixelShuffle(2),
#             CP_Attention_block(16),
#             Block_updata(16),
#             # nn.Conv2d(16, 16, 3, 1, 1),
#             # nn.PReLU(),
#             nn.Conv2d(16, 3, 7, 1, 3),
#             nn.Tanh()
#
#         )
#
#
#     def forward(self,input,encode_x1,encode_x2,Fusion_data,encode_x3):
#
#         x1 = self.upconv1(torch.cat((Fusion_data,encode_x3),dim=1))
#         x2 = self.upconv2(torch.cat((x1,encode_x2),dim=1))
#         out = self.upconv3(torch.cat((x2,encode_x1),dim=1)) + input
#
#         return  out


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
        # self.input = nn.Sequential(
        #     nn.Conv2d(3,64,3,2,1),
        #     # nn.Conv2d(64, 128, 3, 2, 1),
        #     Block(64),
        #     # nn.PReLU(),
        # )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(64, 256, 3, 2, 1),
        #     # nn.Conv2d(256, 512, 3, 2, 1),
        #     Block(256),
        #     # nn.PReLU(),
        #     )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(256, 1024, 3, 2, 1),
        #     Block(1024),
        #     # nn.PReLU(),
        # )

        # self.up_block = nn.PixelShuffle(2)
        # self.attention1 = CgaAttenaiton()
        # self.attention2 = CgaAttenaiton()
        # self.attention3 = CgaAttenaiton()
        # self.attention4 = CgaAttenaiton()
        self.encode = Encode()
        self.prior = nn.Sequential(
            knowledge_CodeBook(),
            nn.Conv2d(768, 1024, 7, 1, 3),
        )


        self.fsu = PriorFusion(1024)

        self.decode = Decode()



    def forward(self,input): # 1,3,320,320
        # print(input.size())
        x1,x2,x3 = self.encode(input)
        # print(x3.shape) # torch.Size([1, 256, 40, 40])
        # pripr
        # prior = self.prior(input)
        # print((prior.shape)) # torch.Size([1, 768, 40, 40])
        # x4 = self.fsu(x3,prior)
        # print(x4.shape,x2.shape)
        out = self.decode(input,x1,x2,x3)

        return out