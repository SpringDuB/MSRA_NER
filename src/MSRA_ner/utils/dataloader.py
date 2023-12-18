import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, AlbertTokenizer

from src.MSRA_ner.utils.constants import TAG2ID, is_end_tag_id
from src.MSRA_ner.utils.param import Param


class InputFeature(object):
    """
    一个训练样本就是一个InputFeature对象
    """

    def __init__(self, input_ids, label_ids):
        super(InputFeature, self).__init__()
        self.input_ids = input_ids  # list[int]的形式
        self.label_ids = label_ids  # list[int]的形式


def read_examples(data_file_path):
    """
    从文件中读取数据，并加载到内存中
    :param data_file_path:
    :return:
    """
    examples = []
    with open(data_file_path, "r", encoding="utf-8") as reader:
        for line in reader:
            x = line.strip()
            if x:
                y = reader.readline().strip()
                # print(y)
                example = [
                    x.split(" "),
                    y.split(" ")
                ]
                if len(example[0]) == len(example[1]):
                    examples.append(example)
            else:
                break
    print(f"数据量:{len(examples)} -- {data_file_path}")
    return examples


def convert_example_to_feature(examples, tokenizer, max_sequence_length):
    features = []
    for tokens, labels in tqdm(examples):
        # 1. token字符转id
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 2. label名称转label id
        label_ids = [TAG2ID[label] for label in labels]
        if len(token_ids) != len(label_ids):
            continue

        # 3. 构造feature
        token_length = len(token_ids)
        bunks = int(np.ceil(token_length / (max_sequence_length - 2)))
        len_per_features = min(int(token_length / bunks), max_sequence_length - 2)
        start_idx = 0
        while start_idx < token_length:
            end_idx = min(start_idx + len_per_features, token_length)
            while not is_end_tag_id(label_ids[end_idx - 1]):
                end_idx = end_idx - 1  # 如果当前最后一个token对用的标签不是可以结尾的标签，那么end_idx前移一位
                if end_idx <= start_idx:
                    break
            if end_idx <= start_idx:
                # 异常，无法分割数据，直接退出
                break
            cur_token_ids = tokenizer.build_inputs_with_special_tokens(token_ids[start_idx:end_idx], None)
            cur_label_ids = [TAG2ID['O']] + label_ids[start_idx:end_idx] + [TAG2ID['O']]
            feature = InputFeature(input_ids=cur_token_ids, label_ids=cur_label_ids)
            features.append(feature)
            start_idx = end_idx
    return features


class NERDataset(Dataset):
    def __init__(self, features):
        super(NERDataset, self).__init__()
        self.features = features  # list(InputFeature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]  # token ids、label ids


class NERDataLoader(object):
    def __init__(self, tokenizer_class, param: Param):
        super(NERDataLoader, self).__init__()
        self.param = param
        # 恢复token的解析器
        self.tokenizer = tokenizer_class.from_pretrained(param.tokenizer_path)

    def collate_fn(self, batch):
        # 最终返回的应该是tensor对象，并且包含: token ids、attention mask、label ids
        batch_token_ids = []  # [N,T] 有填充 LongTensor
        batch_attention_mask = []  # [N,T] 有填充 FloatTensor
        batch_label_ids = []  # [N,T] 有填充 LongTensor
        max_seq_length = int(max([len(b.input_ids) for b in batch]))

        for feature in batch:
            input_ids = copy.deepcopy(feature.input_ids)
            label_ids = copy.deepcopy(feature.label_ids)
            input_length = len(input_ids)
            if input_length < max_seq_length:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_seq_length - input_length)
                label_ids = label_ids + [TAG2ID['O']] * (max_seq_length - input_length)
            attention_mask = np.asarray([0.0] * max_seq_length)
            attention_mask[:input_length] = 1.0
            batch_token_ids.append(input_ids)
            batch_label_ids.append(label_ids)
            batch_attention_mask.append(list(attention_mask))

        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.float)
        batch_label_ids = torch.tensor(batch_label_ids, dtype=torch.long)

        return batch_token_ids, batch_attention_mask, batch_label_ids

    def get_features(self, task: str):
        _data_path = self.param.data_path
        _max_seq_length = self.param.max_sequence_length

        cache_path = os.path.join(_data_path, f"{task}.cache.{_max_seq_length}")
        if os.path.exists(cache_path):
            print(f"从缓存文件对象中加载数据:{cache_path}")
            features = torch.load(cache_path, map_location='cpu')
        else:
            print("=" * 100)
            print(f"开始构造初始的{task}数据集~~~")
            examples = read_examples(os.path.join(_data_path, f"{task}.txt"))  # a. 遍历文本得到x和y
            features = convert_example_to_feature(examples, self.tokenizer, _max_seq_length)  # b. 将文本x转换为token ids
            torch.save(features, cache_path)
        print(f"总样本数目:{len(features)}")
        return features

    def get_dataloader(self, task: str) -> DataLoader:
        # 1. 构造Dataset
        dataset = NERDataset(self.get_features(task))

        # 2. 构造DataLoader
        if task == 'train':
            batch_size = self.param.batch_size
            sampler = RandomSampler(dataset)  # 用在批次数据获取的时候，一个批次具体包含哪些index对应的数据
        elif task == 'val':
            batch_size = self.param.batch_size * 2
            sampler = SequentialSampler(dataset)
        elif task == 'test':
            batch_size = self.param.batch_size * 2
            sampler = SequentialSampler(dataset)
        else:
            raise ValueError(f"参数task异常，可选值:[train, val, test]， 当前值:{task}")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn
        )
        return dataloader

    