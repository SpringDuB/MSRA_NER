import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DebertaModel, DebertaTokenizer  # pip install transformers
from transformers.modeling_outputs import BaseModelOutput

from . import LanguageModel
from ...utils.param import Param


class DebertaLanguageModel(LanguageModel):
    tokenizer_class = DebertaTokenizer

    def __init__(self, param: Param):
        super(DebertaLanguageModel, self).__init__(
            output_dim=param.deberta_config.hidden_size,
            vocab_size=param.lm_args['vocab_size']
        )

        _args = param.lm_args
        _cfg = param.deberta_config
        try:
            # 迁移模型
            pretrained_model_name_or_path = _args.get('fine_tune_root_dir')
            if pretrained_model_name_or_path is None:
                self.bert = DebertaModel(config=_cfg)
            else:
                self.bert = DebertaModel.from_pretrained(
                    config=_cfg,
                    pretrained_model_name_or_path=pretrained_model_name_or_path
                )
                print("Deberta模型参数迁移成功!")
        except Exception as e:
            print(f"迁移Deberta模型异常:{e}")
            self.bert = DebertaModel(config=_cfg)

        # 定义各个输出层的置信度，默认是仅使用最后一层的输出
        _layers = _cfg.num_hidden_layers
        self.alpha = nn.Parameter(
            data=(1.0 - F.one_hot(torch.tensor(_layers - 1, dtype=torch.long), _layers)) * -1000.0
        )

    def forward(self, input_ids: Tensor, input_mask: Tensor, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            output_hidden_states=True,  # 是否返回所有层的hidden_states
            return_dict=True  # 以对象的形式返回，默认就是True
        )

        # 提取所有输出层，进行加权合并
        if isinstance(outputs, BaseModelOutput):
            output_hidden_states = torch.stack(outputs.hidden_states[1:])  # [num_layers,N,T,hidden_size]
        else:
            output_hidden_states = torch.stack(outputs[1][1:])  # [num_layers,N,T,hidden_size]
        # [num_layers,N,T,hidden_size] * [num_layers,1,1,1] = [num_layers,N,T,hidden_size]
        alpha = torch.softmax(self.alpha, dim=0)  # [num_layers]
        output_hidden_states = output_hidden_states * alpha[..., None, None, None]
        last_hidden_state = output_hidden_states.mean(dim=0)

        # mask加权输出
        last_hidden_state = last_hidden_state * input_mask[..., None]
        return last_hidden_state  # [N,T,E]
