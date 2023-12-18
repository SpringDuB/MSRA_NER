import torch.nn as nn
from torch import Tensor


class EncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(EncoderModel, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

    def forward(self, input_feat: Tensor, input_mask: Tensor, **kwargs):
        """
        前向过程
        :param input_feat: 输入的特征信息，FloatTensor类型，[N,T,E]
        :param input_mask: mask信息，实际值位置为1，填充位置为0 [N,T]
        :param kwargs:
        :return: [N,T,E]
        """
        raise NotImplementedError("未实现!")

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim
