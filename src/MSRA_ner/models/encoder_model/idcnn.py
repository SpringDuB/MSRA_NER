import torch
import torch.nn as nn
from torch import Tensor

from . import EncoderModel
from ...utils.param import Param


class IDCNNEncoderModel(EncoderModel):

    def __init__(self, input_dim: int, param: Param, *args, **kwargs):
        super(IDCNNEncoderModel, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim
        )
        dilations = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}
        ]

        _args = param.encoder_args
        self.num_block = _args['idcnn_num_block']
        _kernel_size = _args['idcnn_kernel_size']
        layers = []
        for i in range(len(dilations)):
            dilation = dilations[i]["dilation"]  # 获取膨胀系数
            # 卷积操作：将一个样本的多个通道的特征值合并成一个通道的特征值
            # (N, C_{in}, L_{in}) --> (N, C_{out}, L_{out})
            # 原始文本格式: [N,T,E] --> [N,E,T]  表示有E个通道，每个卷积的每个窗口就是将k个时刻的k*E个特征值合并成一个
            # noinspection PyTypeChecker
            single_block = nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=_kernel_size,
                dilation=dilation,
                padding=_kernel_size // 2 + dilation - 1
            )
            layers.append(
                nn.Sequential(
                    single_block,
                    nn.ReLU(),
                    LayerNorm(input_dim)
                )
            )
        self.block = nn.Sequential(*layers)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.block_norms = nn.ModuleList([LayerNorm(input_dim) for _ in range(self.num_block)])

    def forward(self, input_feat: Tensor, input_mask: Tensor, **kwargs):
        z = torch.permute(input_feat, dims=(0, 2, 1))  # [N,T,E] -> [N,E,T]
        for i in range(self.num_block):
            z_ = self.block(z)  # [N,E,T] -> [N,E,T]
            z = self.gelu(z + z_)
            z = self.block_norms[i](z)
        z = torch.permute(z, dims=(0, 2, 1))  # [N,E,T] -> [N,T,E]

        z = self.gelu(input_feat + z)
        z = self.norm(z)
        return z


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, features, 1))  # [N,E,T]
        self.beta = nn.Parameter(torch.zeros(1, features, 1))  # [N,E,T]
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
