import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.constants import ID2TAG
from utils.metrics import f1_score, accuracy_score, classification_report
from utils.param import Param
from utils.running_utils import RunningAverage


@torch.no_grad()
def evaluate_epoch(model: nn.Module, data_loader, param: Param, mark='Val', verbose=True):
    device = param.device
    model.eval()

    # 遍历数据，获取loss和预测标签值
    loss_avg = RunningAverage()
    pred_tags = []  # 预测标签 list[str]
    true_tags = []  # 真实标签 list[str]
    for input_ids, attention_mask, label_ids in tqdm(data_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)
        batch_size, max_len = label_ids.size()

        # loss & inference
        scores, loss, batch_pred_label_ids = model(input_ids, attention_mask, labels=label_ids, return_output=True)
        if param.n_gpu > 1:
            loss = loss.mean()  # 多gpu运行的时候模型返回的是一个tensor[N]结构
        loss_avg.update(loss.item())

        # 恢复真实标签的信息
        for i in range(batch_size):
            real_len = int(attention_mask[i].sum().item())
            real_label_ids = label_ids[i][:real_len].to('cpu').numpy()
            pred_label_ids = batch_pred_label_ids[i][:real_len].to('cpu').numpy()
            for real_label_id, pred_label_id in zip(real_label_ids, pred_label_ids):
                true_tags.append(ID2TAG[real_label_id])
                pred_tags.append(ID2TAG[pred_label_id])
    assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'

    # 开始计算评估指标
    metrics = {
        'loss': loss_avg(),
        'f1': f1_score(true_tags, pred_tags),
        'accuracy': accuracy_score(true_tags, pred_tags)
    }
    msg = f"-{mark} metrics: {'; '.join(map(lambda t: f'{t[0]}: {t[1]:.3f}', metrics.items()))}"
    logging.info(msg)
    print(msg)
    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics
