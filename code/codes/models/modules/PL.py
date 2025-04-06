

import torch
from thop import profile

from  DenoisingNAFNet_arch import ConditionalNAFNet

# torch.Size([2, 16, 128, 128]) torch.Size([2, 16, 128, 128]) torch.Size([2])
model = ConditionalNAFNet(16,64,enc_blk_nums=[1, 1, 1, 28],middle_blk_num=1,dec_blk_nums=[1, 1, 1, 1])
input1 = torch.randn(1, 16, 128, 128)
flops, params = profile(model, inputs=(input1,input1,[2] ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')



"""
network_G:
  which_model: ConditionalNAFNet
  setting:
    img_channel: 16
    width: 64
    enc_blk_nums: [1, 1, 1, 28]
    middle_blk_num: 1
    dec_blk_nums: [1, 1, 1, 1]
"""