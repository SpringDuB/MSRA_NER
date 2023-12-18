import torch.nn as nn
from torch import Tensor
from transformers import BertTokenizer

from ._base import LanguageModel
from ...utils.param import Param


class Word2VecLanguageModel(LanguageModel):
    tokenizer_class = BertTokenizer

    def __init__(self, param: Param):
        super(Word2VecLanguageModel, self).__init__(
            output_dim=param.lm_args['embedding_dim'],
            vocab_size=param.lm_args['vocab_size']
        )
        _args = param.lm_args
        self.embd_layer = nn.Embedding(
            num_embeddings=_args['vocab_size'],
            embedding_dim=_args['embedding_dim']
        )

    def forward(self, input_ids: Tensor, input_mask: Tensor, **kwargs):
        z = self.embd_layer(input_ids)  # [N,T] -> [N,T,E]
        z = z * input_mask[..., None]
        return z
