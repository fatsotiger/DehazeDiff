import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
from models.UNet_Model import Encode,Decode
from .module_util import (
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual, Identity)


# class UNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#
#
#         self.encoder = Encode()
#         self.encoder.eval()
#         # ckpt = torch.load("./models/ckpt.pt")
#         # self.model.load_state_dict(ckpt)
#
#         self.decoder = Decode()
#         self.decoder.eval()
#
#
#     def check_image_size(self, x, h, w):
#         s = int(math.pow(2, self.depth))
#         mod_pad_h = (s - h % s) % s
#         mod_pad_w = (s - w % s) % s
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def encode(self, x):
#
#         x1,x2,fusion,x3 = self.encoder(x)
#
#         return x1,x2,fusion,x3
#
#     def decode(self, x,x1,x2,fusion,x3):
#
#         output = self.decoder(x,x1,x2,fusion,x3)
#         return output
#
#     def forward(self, x):
#         x1,x2,fusion,x3 = self.encode(x)
#         out = self.decode(x,x1,x2,fusion,x3)
#
#         return out


class UNet(nn.Module):
    def __init__(self,in_ch=3, out_ch=3, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()



        self.encoder = Encode()
        # self.encoder.eval()
        # ckpt = torch.load("./models/ckpt.pt")
        # self.model.load_state_dict(ckpt)

        self.decoder = Decode()
        # self.apply(self._weight_init)
        # self.decoder.eval()


    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    # def encode(self, x):
    #
    #     x1,x2,x3,channalpre = self.encoder(x)
    #
    #     return x1,x2,x3,channalpre
    #
    # def decode(self, x,x1,x2,x3,channalpre):
    #
    #     output = self.decoder(x,x1,x2,x3,channalpre)
    #     return output
    #
    # def forward(self, x):
    #     x1,x2 ,x3,channalpre = self.encode(x)
    #     out = self.decode(x,x1,x2,x3,channalpre)

        return out
    def encode(self, x):

        x1,x2,fusion,x3 = self.encoder(x)

        return x1,x2,fusion,x3

    def decode(self, x,x1,x2,fusion,x3):

        output = self.decoder(x,x1,x2,fusion,x3)
        return output

    def forward(self, x):
        x1,x2,fusion ,x3 = self.encode(x)
        out = self.decode(x,x1,x2,fusion,x3)

        return out



