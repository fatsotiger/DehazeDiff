

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.UNet_Model.Attention import CgaAttenaiton_updata






class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # 查询、键和值卷积
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C x N
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N

        # 计算注意力图
        # print(proj_query.shape, proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N

        # 计算自注意力特征
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)  # B x C x W x H

        # 融合输入和自注意力特征
        out = self.gamma * out + x
        return out
class PriorFusion(nn.Module):
    def __init__(self,dim): # dim = C_x + C_y
        super(PriorFusion,self).__init__()
        # self.SA = SelfAttention(dim)
        self.Gca = CgaAttenaiton_updata(dim*2)
        # self.conv = nn.Conv2d(dim*2, dim, 1, bias=True)
        # self.conv_p = nn.Conv2d(dim*2 , dim*2, 1, bias=True)

        # self.conv_out = nn.Conv2d(dim * 2, dim, 7, stride=1,padding=3,bias=True)

    def forward(self,x): # x = out y =  prior
        #initial = self.conv_p(x)
        # print(initial.shape)
        out  = self.Gca(x)
        # out = self.conv(torch.cat((initial + w * x + (1-w)*y,out),dim=1)) # out


        # x = self.SA(input)

        return out