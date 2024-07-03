# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM,local_vssm_tiny_search,local_vssm_tiny,local_vssm_small_search,local_vssm_small
logger = logging.getLogger(__name__)
#模型的配置参数 zero_head表示是否使用零初始化的头部
class MambaUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        #模型的主要架构和参数 
        self.mamba_unet = VSSM(
                                patch_size=config.MODEL.VSSM.PATCH_SIZE,
                                in_chans=config.MODEL.VSSM.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.VSSM.EMBED_DIM,
                                depths=config.MODEL.VSSM.DEPTHS,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.VSSM.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                )
    #前向传播
    def forward(self, x):
        #如果输入的图片是单通道的，就复制成三通道，以匹配所需的三通道输入(rgb)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        #得到输出
        outs = self.mamba_unet(x)
        return outs
    
    #加载预训练模型
    def load_from(self, config):
        #加载预训练模型
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        #如果有预训练模型，就加载预训练模型
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            #判断是否有gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #加载预训练模型
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            #如果预训练模型中没有model这个key，就说明是分开保存的模型，需要进行处理
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                #将pretrained_dict中所有键从第17个字符开始的部分作为新的键名
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                #遍历修改后的pretrained_dict字典中的所有键，如果键中包含output，就删除这个键
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                #加载权重到模型mamba_unet中
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            #如果预训练模型中有model这个key，就说明是整体保存的模型，直接加载即可
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")
            #得到模型的参数字典
            model_dict = self.mamba_unet.state_dict()
            #复制预训练模型的参数字典
            full_dict = copy.deepcopy(pretrained_dict)
            #它包含字符串"layers."，则意味着这个权重属于模型的某个层
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])#得到当前层的编号
                    current_k = "layers_up." + str(current_layer_num) + k[8:]#将当前层的编号加到键名的前面
                    full_dict.update({current_k:v})
            #代码检查full_dict中的每个键是否存在于模型的权重字典model_dict中。如果存在，它会比较full_dict中的权重形状和model_dict中权重的形状是否一致。如果不一致，就删除full_dict中的这个键
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")