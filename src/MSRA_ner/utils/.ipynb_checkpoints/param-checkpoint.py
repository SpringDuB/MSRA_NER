"""
定义入参对象
"""
import argparse
import copy
import json
import logging
import os

from transformers import BertConfig, AlbertConfig, DebertaConfig, RobertaConfig

# noinspection SpellCheckingInspection,PyTypeChecker
from src.MSRA_ner.utils.constants import TAGS


# noinspection PyTypeChecker
class Param(object):
    def __init__(self, params: dict = None):
        super(Param, self).__init__()
        if params is None:
            params = {
                'device': 'cuda',
                'summary_log_dir': r".\outputs\summary",
                'resume': False,
                'freeze_lm': False
            }

        self.params = params  # 临时保存一下

        self.resume = params['resume']
        self.freeze_lm = params['freeze_lm']

        self.device = params['device']
        self.n_gpu = 1

        self.summary_log_dir = params['summary_log_dir']
        os.makedirs(self.summary_log_dir, exist_ok=True)
        self.model_save_dir = params.get('model_save_dir', r"outputs/models/ckpt")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.data_path = params.get('data_path', r"/mnt/workspace/nlp/MSRA_ner/datas/MSRA_Pro")
        self.max_sequence_length = 64

        self.batch_size = 64
        self.epoch_num = 20  # 总训练epoch数量
        self.gradient_accumulation_steps = 1  # 训练过程中，间隔多少个批次进行一次参数更新
        self.max_grad_norm = 2.0  # 梯度截断值
        self.warmup_prop = 0.1  # warmup的参数
        self.warmup_schedule = "warmup_cosine"  # warmup参数

        # 1. 第一层模型的相关参数
        self.lm_name = params.get('lm_name', 'BertLanguageModel')
        self.lm_args = {
            'weight_decay': params.get('lm_weight_decay', 0.001),
            'lr': params.get('lm_lr', 0.0001),

            'vocab_size': 100,
            'embedding_dim': 768,
        }
        if 'fine_tune_root_dir' in params and os.path.exists(params['fine_tune_root_dir']):
            # r'C:\Users\HP\dataroot\models\bert-base-chinese'
            self.lm_args['fine_tune_root_dir'] = params['fine_tune_root_dir']  # 迁移模型路径字符串
        self.lm_cfg = None
        if self.lm_name == 'BertLanguageModel':
            self.lm_cfg = self.bert_config
            self.lm_args['vocab_size'] = self.lm_cfg.vocab_size
            self.lm_args['embedding_dim'] = self.lm_cfg.hidden_size
        elif self.lm_name == 'DebertaLanguageModel':
            self.lm_cfg = self.deberta_config
            self.lm_args['vocab_size'] = self.lm_cfg.vocab_size
            self.lm_args['embedding_dim'] = self.lm_cfg.hidden_size
        elif self.lm_name == 'xxx':
            # TODO: 针对每个迁移的都进行对应的覆盖即可
            pass

        # 2. 第二层特征融合模型的相关参数
        self.encoder_name = "RTransformerEncoderModel"
        self.encoder_args = {
            'weight_decay': 0.001,
            'lr': 0.001,

            'hidden_size': 128,
            'num_layers': 2,
            'idcnn_num_block': 4,
            'idcnn_kernel_size': 3
        }

        # 3. 第三层决策输出模型的相关参数
        self.classify_name = "SoftmaxSeqClassifyModel"
        # self.classify_name = "CRFSeqClassifyModel"
        self.classify_args = {
            'weight_decay': 0.001,
            'lr': 0.001,
            'crf_lr': 0.1,
            'num_classes': len(TAGS)
        }

    @property
    def bert_config(self) -> BertConfig:
        _dir = self.lm_args.get('fine_tune_root_dir')
        if _dir is None:
            return BertConfig(**self.params)  # TODO: 待完善
        else:
            return BertConfig.from_pretrained(_dir)

    @property
    def roberta_config(self) -> RobertaConfig:
        _dir = self.lm_args.get('fine_tune_root_dir')
        if _dir is None:
            return RobertaConfig(**self.params)  # TODO: 待完善
        else:
            return RobertaConfig.from_pretrained(_dir)

    @property
    def albert_config(self) -> AlbertConfig:
        _dir = self.lm_args.get('fine_tune_root_dir')
        if _dir is None:
            return AlbertConfig(**self.params)  # TODO: 待完善
        else:
            return AlbertConfig.from_pretrained(_dir)

    @property
    def deberta_config(self) -> DebertaConfig:
        _dir = self.lm_args.get('fine_tune_root_dir')
        if _dir is None:
            return DebertaConfig(relative_attention=True, **self.params)  # TODO: 待完善
        else:
            return DebertaConfig.from_pretrained(_dir)

    @property
    def tokenizer_path(self) -> str:
        _dir = self.lm_args.get('fine_tune_root_dir')
        if _dir is None:
            _dir = r'C:\Users\HP\dataroot\models\bert-base-chinese'
        return _dir  # TODO:待完善

    def to_dict(self) -> dict:
        _params = copy.deepcopy(self.__dict__)
        del _params['params']
        lm_cfg = _params.get('lm_cfg')
        if lm_cfg is not None:
            _params['lm_cfg'] = lm_cfg.to_dict()
        return _params

    def save(self):
        param_path = os.path.join(self.model_save_dir, "..", "param.json")
        with open(param_path, "w", encoding='utf-8') as writer:
            json.dump(self.params, writer)

    @staticmethod
    def load(param_dir) -> 'Param':
        param_path = os.path.join(param_dir, "param.json")
        with open(param_path, "r", encoding='utf-8') as reader:
            params = json.load(reader)
        return Param(params)


def build_params(resume=False):
    """
    构造参数对象
    :param resume: 是否强制进行参数恢复
    :return:
    """
    # noinspection DuplicatedCode
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="运行设备:cpu,cuda0,cuda1...", required=False, type=str, default='cuda')
    parser.add_argument("-summary_log_dir", help="日志路径", required=False, type=str, default=r"/mnt/workspace/nlp/MSRA_ner/src/MSRA_ner/outputs/summary")
    parser.add_argument("-model_save_dir", help="模型存储路径", required=False, type=str, default=r"/mnt/workspace/nlp/MSRA_ner/src/MSRA_ner/outputs/models/ckpt")
    parser.add_argument("-data_path", help="数据文件路径", required=False, type=str, default=r"/mnt/workspace/nlp/MSRA_ner/datas/MSRA_Pro")
    parser.add_argument("-fine_tune_root_dir", help="基础语言模型的迁移路径", required=False, type=str,
                        default=r'/mnt/workspace/nlp/models/bert-base-chinese')
    parser.add_argument("-vocab_size", help="给定词汇表大小，但是当进行模型迁移的时候，该参数无效.", required=False, type=int, default=30522)
    parser.add_argument("-lm_name", help="给定语言模型的基地,可选:BertLanguageModel...", required=False, type=str,
                        default='BertLanguageModel')
    parser.add_argument("-resume", action='store_const', const=True, default=False, help='模型恢复训练，要求模型输出路径必须是一致的。')
    parser.add_argument("-freeze_lm", action='store_const', const=True, default=False,
                        help='给定冻结基础的语言模型参数，冻结后这部分参数不参与模型训练。')
    args = parser.parse_args()
    print("args: '{}'".format(args))
    print(type(args))
    print(args.device)

    param = Param(vars(args))
    if args.resume or resume:
        logging.info("模型恢复继续训练，先恢复模型参数....")
        param = Param.load(os.path.join(param.model_save_dir, ".."))
        param.resume = True
    # 模型参数保存
    param.save()
    return param
