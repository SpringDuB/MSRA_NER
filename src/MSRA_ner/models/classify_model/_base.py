from typing import Optional

import torch.nn as nn
from torch import Tensor


class ClassifyModel(nn.Module):
    def __init__(self, input_dim, num_classes, **kwargs):
        super(ClassifyModel, self).__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes

    def forward(self, input_feat: Tensor, input_mask: Tensor, labels: Optional[Tensor] = None, return_output=False,
                **kwargs):
        """
        前向过程
        :param input_feat: 输入的特征信息，FloatTensor类型，[N,??,E]
        :param input_mask: mask信息，实际值位置为1，填充位置为0
        :param labels: 计算loss用的预测类别标签， [N,T] LongTensor
        :param return_output: 是否返回最终预测结果
        :param kwargs:
        :return: [N,??,num_class] loss 预测结果
        """
        raise NotImplementedError("未实现!")

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes
