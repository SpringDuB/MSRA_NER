import torch.nn as nn
from torch import Tensor


class LanguageModel(nn.Module):
    tokenizer_class = None

    def __init__(self, output_dim, vocab_size, **kwargs):
        super(LanguageModel, self).__init__()
        self._output_dim = output_dim
        self._vocab_size = vocab_size

    def freeze_model(self):
        print(f"当前语言模型不支持参数冻结:{type(self)}")

    def forward(self, input_ids: Tensor, input_mask: Tensor, **kwargs):
        """
        前向过程
        :param input_ids: 输入的token id信息，LongTensor类型，[N,T]
        :param input_mask: mask信息，实际值位置为1，填充位置为0，[N,T] FloatTensor
        :param kwargs:
        :return: [N,T,E]
        """
        raise NotImplementedError("未实现!")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
