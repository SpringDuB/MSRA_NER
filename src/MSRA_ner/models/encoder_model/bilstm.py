import torch.nn as nn
from torch import Tensor

from ._base import EncoderModel
from ...utils.param import Param


class BiLSTMEncoderModel(EncoderModel):
    def __init__(self, input_dim: int, param: Param, *args, **kwargs):
        super(BiLSTMEncoderModel, self).__init__(
            input_dim=input_dim,
            output_dim=input_dim
        )
        _args = param.encoder_args
        _hidden_size = int(max(input_dim * 0.5, _args['hidden_size']))
        self.lstm_layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=_hidden_size,
            num_layers=_args['num_layers'],
            bidirectional=True,
            batch_first=True
        )
        self.ffn_layer = nn.Sequential(
            nn.Linear(_hidden_size * 2, _hidden_size * 8),
            nn.GELU(),
            nn.Linear(_hidden_size * 8, input_dim)
        )
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(normalized_shape=input_dim)

    def forward(self, input_feat: Tensor, input_mask: Tensor, **kwargs):
        # 1. LSTM转换: [N,T,E] -> [N,T,hidden_size*2]
        # z1, _ = self.lstm_layer(input_feat)
        max_length = input_feat.shape[1]
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            input_feat, lengths=input_mask.sum(1).long(), batch_first=True, enforce_sorted=False
        )
        output_packed_sequence, _ = self.lstm_layer(packed_sequence)
        z1, _ = nn.utils.rnn.pad_packed_sequence(output_packed_sequence, batch_first=True, total_length=max_length)

        # 2. FFN进一步的特征提取:[N,T,hidden_size*2] -> [N,T,E]
        z2 = self.ffn_layer(z1)

        # 3. 做一个残差结构 [N,T,E]
        z3 = self.gelu(input_feat + z2)

        # 4. norm的处理 [N,T,E]
        z4 = self.norm(z3)
        return z4
