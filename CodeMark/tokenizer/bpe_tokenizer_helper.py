import os
import torch
import youtokentome as yttm
import json
import torch.nn.functional as F
from .base_tokenizer_helper import BaseTokenizerHelper
from typing import List, Union


def train_bpe(train_data_path: str, bpe_path: str):
    codes = []
    with open(file=train_data_path, mode='r', encoding='utf-8') as f:
        codes.extend(json.load(f))
    print('load train dataset')
    bpe_train_data = []
    for code in codes:
        for _, subgraph in code['subgraphs'].items():
            for node in subgraph['nodes']:
                bpe_train_data.append(node[2])
    temp_path = 'temp_bpe_train.txt'
    with open(temp_path, "w") as f:
        for data in bpe_train_data:
            f.write(data + "\n")
    yttm.BPE.train(data=temp_path, vocab_size=5000, model=bpe_path)
    os.remove(temp_path)


class BPETokenizerHelper(BaseTokenizerHelper):
    def __init__(self, bpe_path):
        super(BPETokenizerHelper, self).__init__()
        self.bpe = yttm.BPE(model=bpe_path)

    # def get_bpe(self):
    #     return self.bpe
    def encode(self, sentence: List[str]) -> Union[List[List[int]], List[List[str]]]:
        return self.bpe.encode(sentence)

    def decode(self):
        pass

    def get_pad_id(self):
        return self.bpe.subword_to_id('<PAD>')


if __name__ == '__main__':
    bpe_tokenizer = BPETokenizerHelper('/mnt/members/liwei/Code-Watermark/variable-watermark/bpe.model')
    # code = ['String   basePath   =   " /tmp/ "']
    # print(bpe_tokenizer.get_bpe().encode(code))
    # print([bpe_tokenizer.get_bpe().encode(token) for token in 'String   basePath   =   " /tmp/ "'.split(' ') if token != ''])
    # print(bpe_tokenizer.get_bpe().id_to_subword(57))
    # print(F.one_hot(torch.tensor([1,2,3]),num_classes=5))
    print(bpe_tokenizer.get_bpe().id_to_subword(0))
