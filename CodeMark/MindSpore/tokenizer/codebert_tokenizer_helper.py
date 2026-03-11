from .base_tokenizer_helper import BaseTokenizerHelper
from typing import List, Union
from transformers import RobertaTokenizerFast

# 0 1 2 is bos_token_id, pad_token_id, eos_token_id
class CodeBERTTokenizerHelper(BaseTokenizerHelper):
    def __init__(self, model_path: str):
        super(CodeBERTTokenizerHelper, self).__init__()
        self.codebert_tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.pad_token_id = self.codebert_tokenizer.pad_token_id
        self.bos_token_id = self.codebert_tokenizer.bos_token_id
        self.eos_token_id = self.codebert_tokenizer.eos_token_id


    def encode(self, sentence: List[str]) -> List[int]:
        temp = self.codebert_tokenizer.encode(sentence)
        temp = temp[1:-1]
        return temp

    def decode(self, tok_ids: List[int])->List[str]:
        return self.codebert_tokenizer.decode(tok_ids)

    def get_pad_id(self):
        return self.codebert_tokenizer.pad_token_id
