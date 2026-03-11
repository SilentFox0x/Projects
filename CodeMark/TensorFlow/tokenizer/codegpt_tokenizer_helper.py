from .base_tokenizer_helper import BaseTokenizerHelper
from typing import List, Union
from transformers import GPT2Tokenizer


class CodeGPTTokenizerHelper(BaseTokenizerHelper):
    def __init__(self, model_path: str):
        super(CodeGPTTokenizerHelper, self).__init__()
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        self.pad_token_id = self.gpt_tokenizer.pad_token_id
        self.bos_token_id = self.gpt_tokenizer.bos_token_id
        self.eos_token_id = self.gpt_tokenizer.eos_token_id

    # def get_bpe(self):
    #     return self.bpe
    def encode(self, sentence: List[str]) -> Union[List[List[int]], List[List[str]]]:
        return self.gpt_tokenizer.encode(sentence)

    def decode(self):
        pass

    def get_pad_id(self):
        return self.gpt_tokenizer.pad_token_id
