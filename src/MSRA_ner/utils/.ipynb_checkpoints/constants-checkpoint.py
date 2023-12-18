from itertools import chain

# EN_DICT = {
#     '疾病和诊断': 'DIS',
#     '手术': 'OPE',
#     '解剖部位': 'POS',
#     '药物': 'MED',
#     '影像检查': 'SCR',
#     '实验室检验': 'LAB'
# }
EN_DICT = {
    '地名': 'LOC',
    '人名': 'PER',
    '组织名': 'ORG',
}

TAGS = ['O']
"""表示标签开始和结束，用于CRF"""
START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
TAGS.extend([START_TAG, END_TAG])
NER_ENTITY_START_ID = len(TAGS)  # 实际实体标签起始下标是3
TAGS.extend(
    list(chain(*map(lambda tag: [f"B-{tag}", f"I-{tag}"], EN_DICT.values())))
)

ID2TAG = dict(enumerate(TAGS))
TAG2ID = {v: k for k, v in ID2TAG.items()}

# def is_end_tag_id(tag_id):
#     tag = ID2TAG[tag_id]
#     if tag == 'O':
#         return True
#     if tag.startswith("S-"):
#         return True
#     if tag.startswith("E-"):
#         return True
#     return False
def is_end_tag_id(tag_id):
    tag = ID2TAG[tag_id]
    if tag == 'O':
        return True
    return False
