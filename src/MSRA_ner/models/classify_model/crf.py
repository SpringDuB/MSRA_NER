# -!-coding:utf-8-!-
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.MSRA_ner.models.classify_model._base import ClassifyModel
from src.MSRA_ner.utils import constants
from src.MSRA_ner.utils.constants import TAG2ID
from src.MSRA_ner.utils.param import Param

"""表示标签开始和结束，用于CRF"""
START_TAG = constants.START_TAG
END_TAG = constants.END_TAG


def log_sum_exp(tensor: torch.Tensor,
                dim: int = -1,
                keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


class CRFLayer(nn.Module):
    def __init__(self, tag2idx):
        super(CRFLayer, self).__init__()
        tag_size = len(tag2idx)

        # transition[i][j] means transition probability from j to i
        # 每一列表示一个转换置信度，eg：第i行第j列的值就表示从状态j转换到状态i的置信度
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size), requires_grad=True)
        self.tag2idx = tag2idx
        # 重置transition参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置transition参数
        """
        nn.init.xavier_normal_(self.transition)
        # initialize START_TAG, END_TAG probability in log space
        # 从i到start和从end到i的score都应该为负
        self.transition.detach()[self.tag2idx[START_TAG], :] = -10000
        self.transition.detach()[:, self.tag2idx[END_TAG]] = -10000

    def forward(self, feats, mask):
        """求total scores of all the paths
        Arg:
          feats: tag概率分布. (seq_len, batch_size, tag_size) [T,N,M] 全连接的输出置信度
          mask: 填充. (seq_len, batch_size) 样本的填充情况
        Return:
          scores: (batch_size, )
        """
        # 获取得到序列长度T、批次大小N以及标签数目M
        seq_len, batch_size, tag_size = feats.size()
        # initialize alpha to zero in log space [N,M] 保存每个样本当前时刻处于各个状态的置信度
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
        # alpha in START_TAG is 1 将起始标签的置信度设置为比较大的初始值
        alpha[:, self.tag2idx[START_TAG]] = 0

        # 取当前step的emit score，获取当前时刻对应的token
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # emit_score is the same regardless of current_tag, so we broadcast along current_tag
            # 获取得到观测值到当前各个状态值的预测置信度(特征函数s对应的最终置信度)
            # N个文本，每个文本属于M个类别的置信度 [N,M]->[N,M,1]
            emit_score = feat.unsqueeze(-1)  # (batch_size, tag_size, 1) # [N,M]->[N,M,1]
            # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
            # M个类别到M个类别的概率转换置信度 [M,M] -> [1,M,M]
            transition_score = self.transition.unsqueeze(0)  # (1, tag_size, tag_size) [M,M] -> [1,M,M]
            # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
            # N个文本，每个文本到M个类别的置信度 [N,M] --> [N,1,M]
            alpha_score = alpha.unsqueeze(1)  # (batch_size, 1, tag_size) [N,M] --> [N,1,M]

            # 基于上一个时刻各个状态的置信度 + 转移置信度 + 发射置信度 --> 当前时刻各个状态的置信度
            # [N,1,M] + [1,M,M] -> [N,M,M] ===> N个样本， 每个样本从M个状态到M个状态的分别的置信度
            # [N,M,M] + [N,M,1] -> [N,M,M] ===> N个样本， 每个样本从M个状态到M个状态的分别的置信度 + N个样本每个样本属于M个类别的置信度
            alpha_score = alpha_score + transition_score + emit_score  # (batch_size, tag_size, tag_size)
            # log_sum_exp along current_tag dimension to get next_tag alpha
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            # 累加每次的alpha
            alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * torch.logical_not(mask_t)  # (batch_size, tag_size)
        # arrive at END_TAG
        alpha = alpha + self.transition[self.tag2idx[END_TAG]].unsqueeze(0)  # (batch_size, tag_size)

        return log_sum_exp(alpha, -1)  # (batch_size, )

    def score_sentence(self, feats, tags, mask):
        """求gold score 求解的是log(pi)的分子部分
        Arg:
          feats: (seq_len, batch_size, tag_size)
          tags: (seq_len, batch_size)
          mask: (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_zeros(batch_size)
        # (seq_len + 1, batch_size) 在实际标签的最前面加一个初始的状态<START_TAG>
        tags = torch.cat([tags.new_full((1, batch_size), fill_value=self.tag2idx[START_TAG]), tags], 0)

        # 取一个step
        for t, feat in enumerate(feats):
            # (batch_size,) 当前时刻t每个样本实际类别对应的预测置信度(线性转换部分)
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])
            # (batch_size,) 从上一个时刻的状态转换到当前时刻状态的状态转移置信度
            transition_score = torch.stack([self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])

            # 合并置信度 当前时刻预测为真实标签的置信度
            step_score = emit_score + transition_score

            # 累加
            scores += step_score * mask[t]

        # 到end的score
        transition_to_end = torch.stack(
            [self.transition[self.tag2idx[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in
             enumerate(tags.transpose(0, 1))])
        scores += transition_to_end
        return scores

    def viterbi_decode(self, feats, mask):
        """维特比算法，解码最佳路径
        :param feats: (seq_len, batch_size, tag_size)
        :param mask: (seq_len, batch_size)
        :return best_path: (seq_len, batch_size)
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize scores in log space [N,M]
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
        scores[:, self.tag2idx[START_TAG]] = 0
        pointers = []

        # forward
        # 取一个step
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # (bat, 1, tag_size) + (1, tag_size, tag_size)
            scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
            # add emit
            scores_t += feat
            pointers.append(pointer)
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_t * mask_t + scores * torch.logical_not(mask_t)
        pointers = torch.stack(pointers, 0)  # (seq_len, batch_size, tag_size)
        scores += self.transition[self.tag2idx[END_TAG]].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )

        # backtracking
        best_path = best_tag.unsqueeze(-1).tolist()  # list shape (batch_size, 1)
        for i in range(batch_size):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                # ptr_t shape (tag_size, )
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # pop first tag
            best_path[i].pop()
            # reverse order
            best_path[i].reverse()
            # mask的预测结果填充
            for _ in range(seq_len - seq_len_i):
                best_path[i].append(0)
        return torch.tensor(best_path)


class CRFSeqClassifyModel(ClassifyModel):
    def __init__(self, input_dim, param: Param, **kwargs):
        super(CRFSeqClassifyModel, self).__init__(
            input_dim,
            param.classify_args['num_classes']
        )

        _args = param.classify_args
        num_classes = _args['num_classes']

        self.fc_layer = nn.Linear(input_dim, num_classes)
        self.crf = CRFLayer(tag2idx=TAG2ID)

    def forward(self, input_feat: Tensor, input_mask: Tensor, labels: Optional[Tensor] = None, return_output=False,
                **kwargs):
        # 1. 前向过程计算预测类别的置信度
        scores = self.fc_layer(input_feat)  # [N,T,num_classes]

        # 2. 计算损失
        loss = None
        crf_feats = torch.permute(scores, dims=(1, 0, 2))  # [N,T,num_classes] -> [T,N,num_classes]
        crf_mask = torch.permute(input_mask, dims=(1, 0))  # [N,T] -> [T,N]
        if labels is not None:
            crf_labels = torch.permute(labels, dims=(1, 0))
            forward_score = self.crf(crf_feats, crf_mask)  # 分母部分
            gold_score = self.crf.score_sentence(crf_feats, crf_labels, crf_mask)  # 分子部分
            loss = forward_score - gold_score  # -log(pi)
            loss = loss.mean()

        # 3. 预测结果
        output = None
        if return_output:
            output = self.crf.viterbi_decode(crf_feats, crf_mask)

        return scores, loss, output
