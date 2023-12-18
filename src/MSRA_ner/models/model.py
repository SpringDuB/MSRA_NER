import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.MSRA_ner.models.classify_model import *
from src.MSRA_ner.models.encoder_model import *
from src.MSRA_ner.models.language_model import *
from src.MSRA_ner.utils.param import Param


class NERTokenClassification(nn.Module):
    def __init__(self, param: Param):
        super(NERTokenClassification, self).__init__()
        self.lm_model: LanguageModel = eval(param.lm_name)(param)
        self.encoder_model = eval(param.encoder_name)(self.lm_model.output_dim, param)
        self.classify_model = eval(param.classify_name)(self.encoder_model.output_dim, param)

    def forward(self, input_ids: Tensor, input_mask: Tensor, labels: Optional[Tensor] = None, return_output=True):
        """
        前向过程
        :param input_ids: token id LongTensor对象 [N,T]
        :param input_mask: mask FloatTensor对象 [N,T]
        :param labels: 当给定labels的时候，会进行loss计算
        :param return_output: 是否返回预测结果，默认为False表示不返回
        :return:
        """
        z = self.lm_model(input_ids, input_mask)
        z = self.encoder_model(z, input_mask)
        z = self.classify_model(z, input_mask, labels=labels, return_output=return_output)
        return z


def restore_model_params(model, param):
    ckpts = os.listdir(param.model_save_dir)
    if len(ckpts) > 0:
        if 'best.pkl' in ckpts:
            ckpt = 'best.pkl'
        else:
            ckpts = sorted(ckpts, key=lambda t: int(t.split(".")[0]))
            ckpt = ckpts[-1]
        ckpt = os.path.join(param.model_save_dir, ckpt)
        logging.info(f"模型恢复文件路径:{ckpt}")
        ckpt = torch.load(ckpt, map_location='cpu')
        missing_keys, _ = model.load_state_dict(ckpt['param'], strict=False)
        if len(missing_keys) > 0:
            logging.warning(f"存在部分参数没有恢复:{missing_keys}")
        logging.info("模型恢复完成!")
        return True
    else:
        return False


def build_model(param: Param):
    model = NERTokenClassification(param)
    if param.resume:
        logging.info("进行模型参数恢复.....")
        restore_model_params(model, param)
    # 参数的冻结实际上并没有限制，但是在真的实现冻结逻辑的时候，我们一般仅冻结整个网络的前面一些层的参数，并且这些层的参数是已经在某些领域中训练好的
    if param.freeze_lm:
        logging.info("冻结基础的语言模型层参数...")
        model.lm_model.freeze_model()

    return model
