from torch import Tensor

from ._base import EncoderModel


class DummyEncoderModel(EncoderModel):
    def __init__(self, input_dim: int, *args, **kwargs):
        super(DummyEncoderModel, self).__init__(input_dim, input_dim)

    def forward(self, input_feat: Tensor, input_mask: Tensor, **kwargs):
        return input_feat
