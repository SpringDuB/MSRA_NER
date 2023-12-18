from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.MSRA_ner.models.classify_model._base import ClassifyModel
from src.MSRA_ner.utils.param import Param


class SoftmaxSeqClassifyModel(ClassifyModel):
    def __init__(self, input_dim, param: Param, **kwargs):
        super(SoftmaxSeqClassifyModel, self).__init__(
            input_dim,
            param.classify_args['num_classes']
        )

        _args = param.classify_args
        num_classes = _args['num_classes']

        self.fc_layer = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self,
                input_feat: Tensor, input_mask: Tensor, labels: Optional[Tensor] = None, return_output=False, **kwargs
                ):
        # 1. 前向过程计算预测类别的置信度
        scores = self.fc_layer(input_feat)  # [N,T,num_classes]

        # 2. 计算损失
        loss = None
        if labels is not None:
            loss_scores = torch.permute(scores, dims=(0, 2, 1))  # [N,T,num_classes] -> [N,num_classes,T]
            loss = self.loss_fn(loss_scores, labels)  # [N,T]
            loss = loss * input_mask.to(loss.dtype)  # [N,T]
            loss = loss.mean()

        # 3. 预测结果
        output = None
        if return_output:
            # 针对每个文本、每个token的num_labels个预测置信度中，选择置信度最高的对应下标作为预测类别id
            # NOTE: 对于同一个样本来讲，连续的两个token之间的预测结果之间是不考虑依赖关系的
            pred_label_ids = torch.argmax(scores, dim=-1)
            output = pred_label_ids * input_mask.to(pred_label_ids.dtype)  # 将填充位置设为为0，0表示特殊类别'O'

        return scores, loss, output
