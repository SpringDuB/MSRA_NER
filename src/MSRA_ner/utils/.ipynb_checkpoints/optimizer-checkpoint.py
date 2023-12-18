import torch.nn as nn

from src.MSRA_ner.utils.optimization import BertAdam
from src.MSRA_ner.utils.param import Param


def build_optimizer(model: nn.Module, param: Param, total_train_batch: int):
    """
    优化器构建
    :param model: 待训练的模型
    :param param: 参数对象
    :param total_train_batch: 总的训练批次数量
    :return:
    """
    # 1. 获取模型参数
    parameter_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # 2. 参数分组
    lm_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith("lm_model.")]
    encoder_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith("encoder_model.")]
    classify_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith("classify_model.")]
    no_decay = ['bias', 'LayerNorm', 'norm', 'layer_norm', 'alpha']
    optimizer_grouped_parameters = [
        # lm_layer + 惩罚系数（L2损失）
        {
            'params': [p for n, p in lm_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': param.lm_args['weight_decay'],  # 惩罚项系数
            'lr': param.lm_args['lr']  # 基础学习率
        },
        # lm_layer
        {
            'params': [p for n, p in lm_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': param.lm_args['lr']  # 基础学习率
        },
        # encoder_layer + l2惩罚
        {
            'params': [p for n, p in encoder_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': param.encoder_args['weight_decay'],
            'lr': param.encoder_args['lr']
        },
        # encoder_layer
        {
            'params': [p for n, p in encoder_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': param.encoder_args['lr']
        },
        # classify_layer + l2惩罚
        {
            'params': [p for n, p in classify_parameters if (not any(nd in n for nd in no_decay)) and 'crf' not in n],
            'weight_decay': param.classify_args['weight_decay'],
            'lr': param.classify_args['lr']
        },
        # classify_layer
        {
            'params': [p for n, p in classify_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': param.classify_args['lr']
        },
        # crf参数学习 --> 一般情况下，不加惩罚性，并且学习率比较大
        {
            'params': [p for n, p in classify_parameters if 'crf' in n],
            'weight_decay': 0.0,
            'lr': param.classify_args['crf_lr']
        }
    ]
    optimizer_grouped_parameters = [ogp for ogp in optimizer_grouped_parameters if len(ogp['params']) > 0]

    # 3. 优化器构建
    from transformers import optimization
    # params, lr=1e-4, warmup=-1, t_total=-1, schedule='warmup_linear',
    #  b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0
    optimizer = BertAdam(
        params=optimizer_grouped_parameters,
        warmup=param.warmup_prop,
        t_total=total_train_batch,  # 给定当前训练中的总的批次数目(可以是近似的，主要影响warmup的执行)
        schedule=param.warmup_schedule,  # 给定warmup学习率变化(前期学习率增大，后期减小)
        max_grad_norm=param.max_grad_norm
    )
    return optimizer
