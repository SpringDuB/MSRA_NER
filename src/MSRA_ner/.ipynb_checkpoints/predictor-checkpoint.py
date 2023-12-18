import torch

from models.model import build_model, restore_model_params
from utils.constants import ID2TAG
from utils.metrics import get_entities
from utils.param import Param


def is_symbols(token):
    symblos = [',', '，', '。', '!', '?']
    for char in token:
        if char in symblos:
            return True
    return False


class Predictor(object):
    def __init__(self, param: Param):
        """
        NOTE:根据需要修改初始化的入参，并且在init代码种完成模型恢复相关逻辑代码
        """
        super(Predictor, self).__init__()

        model = build_model(param).to(param.device)
        if not restore_model_params(model, param):
            raise ValueError("模型参数没有恢复!")
        model = model.eval()
        self.model = model

        self.tokenizer = model.lm_model.tokenizer_class.from_pretrained(param.tokenizer_path)
        self.max_sequence_length = param.max_sequence_length
        print("模型恢复完成!")

    def predict_(self, text, start_idx=0) -> list:
        text_tokens = list(text)  # 原始文本转换为字的token列表

        # 1. token字符转id
        token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        # 2. token添加前后缀
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids, None)
        # 3. 构造mask
        mask = [1.0] * len(token_ids)
        # 4. 模型预测
        _, _, batch_pred_label_ids = self.model(
            torch.tensor([token_ids], dtype=torch.long),
            torch.tensor([mask], dtype=torch.float32),
            return_output=True
        )
        # 恢复真实标签的信息
        pred_label_ids = batch_pred_label_ids[0].to('cpu').numpy()
        pred_tags = []
        for pred_label_id in pred_label_ids:
            pred_tags.append(ID2TAG[pred_label_id])
        pred_entities = set(get_entities(pred_tags, False))

        # 结果封装
        result = []
        for ner in pred_entities:
            result.append({
                'class': ner[0],
                'start': ner[1] + start_idx,
                'end': ner[2] + start_idx,
                'span': ''.join(text_tokens[ner[1]: ner[2] + 1])
            })
        return result

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        print(f"当前待预测文本:{text}")
        i = 0
        seq_len = len(text)
        result = []
        while i < seq_len:
            j = i + self.max_sequence_length - 2
            if j < seq_len:
                k = j
                while not is_symbols(text[k - 1]):
                    k -= 1
                    if k <= i:
                        break
                if k > i:
                    j = k
            sub_text = text[i:j]
            result.extend(self.predict_(sub_text, start_idx=i))
            i = j

        return {
            'code': 0,
            'data': {
                'text': text,
                'ner': result
            }
        }
