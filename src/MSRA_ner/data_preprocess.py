import json
import logging
from sklearn.model_selection import train_test_split

def data_processs(path, save_path):
    texts = []
    labels = []
    labels_dict = dict()
    logging.basicConfig(level=logging.INFO)
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip().split('\t')
            text = line[0]
            label = line[-1]
            if len(label) != 0:
                labels_dict[label] = labels_dict.get(label, len(labels_dict))
                labels.append(label)
            if len(text) != 0:
                texts.append(text)
    f = open(save_path, 'w', encoding='utf-8')
    texts = ''.join(texts).strip().split('。')
    tmp = 0
    for i in range(len(texts)):
        f.write(' '.join(texts[i]).strip() + '\n')
        f.write(' '.join(labels[tmp:tmp+len(texts[i])]).strip() + '\n')
        tmp = tmp + len(texts[i]) + 1
    f.close()

def split_data(path,val_path,split_size):
    x = []
    y = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            x.append(''.join(line.strip().split(' ')))
            y.append(''.join(reader.readline().strip().split(' ')))
    print(len(x), len(y))
    train_text, val_text, train_labels, val_labels = train_test_split(x, y, train_size=split_size)
    f = open(path, 'w', encoding='utf-8')
    for i in range(len(train_text)):
        f.write(' '.join(train_text[i]).strip() + '\n')
        f.write(' '.join(train_labels[i]).strip() + '\n')
    f.close()
    f = open(val_path, 'w', encoding='utf-8')
    for i in range(len(val_text)):
        f.write(' '.join(val_text[i]).strip() + '\n')
        f.write(' '.join(val_labels[i]).strip() + '\n')
    f.close()






if __name__ == '__main__':
    path = r'/mnt/workspace/nlp/MSRA_ner/datas/MSRA/train.txt'
    save_path = r'/mnt/workspace/nlp/MSRA_ner/datas/MSRA_Pro/train.txt'
    # val_path = r'/datas/MSRA_Pro/val.txt'
    split_size = 0.75
    data_processs(path, save_path)
    # split_data(save_path,val_path,split_size)
    path = r'/mnt/workspace/nlp/MSRA_ner/datas/MSRA/test.txt'
    save_path = r'/mnt/workspace/nlp/MSRA_ner/datas/MSRA_Pro/test.txt'
    data_processs(path, save_path)