import os.path
import yaml
import torch
from tokenizer import BPETokenizerHelper, CodeGPTTokenizerHelper, CodeBERTTokenizerHelper
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
import time
from varclr import CLR
from typing import Optional


class Config:
    def __init__(self, config_path, mode: Optional[str] = 'train'):
        with open(file=config_path, mode='r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.__dict__.update(config)
        self.mode = mode

        tokenizer_type = config['tokenizer_type']
        self.__dict__.update(config['tokenizer_config'][tokenizer_type])
        if self.tokenizer_type == 'bpe':
            self.tokenizer = BPETokenizerHelper(self.tokenizer_path)
        elif self.tokenizer_type == 'codegpt':
            self.tokenizer = CodeGPTTokenizerHelper(self.tokenizer_path)
        elif self.tokenizer_type == 'codebert':
            self.tokenizer = CodeBERTTokenizerHelper(self.tokenizer_path)

        if self.use_perplexity:
            self.codegpt = GPT2LMHeadModel.from_pretrained(
                config['tokenizer_config']['codegpt']['tokenizer_path'],
                local_files_only=True, output_hidden_states=True).to(self.device)
            for param in self.codegpt.parameters():
                param.requires_grad = False

        # self.clr = CLR(model_path=config['varclr_path']).to(self.device)
        # for param in self.clr.transformer.parameters():
        #     param.requires_grad = False

        # self.w_loss_weight = self.alpha
        # self.distill_loss_weight = 1 - self.alpha

        self.timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())

        self.logger = self.init_logger(mode=mode)

    def __setitem__(self, key, value):
        raise KeyError('config cannot be changed')

    def print_config(self):
        self.logger.info("=========================== Configuration ===========================")
        attr_dict = vars(self)
        for key, value in attr_dict.items():
            if key == 'codegpt' or key == 'clr':
                continue
            self.logger.info(f"{key}: {value}")
        self.logger.info("=========================== Configuration ===========================")

    def init_logger(self, mode: str):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        handlers = [ch]
        if mode not in ['test', 'play']:
            log_path = os.path.join(os.path.dirname(__file__), f'../logs/{self.timestamp}.txt')
            fh = logging.FileHandler(log_path, mode="w")
            fh.setLevel(logging.INFO)
            handlers.append(fh)
        logging.basicConfig(
            level=logging.INFO,
            handlers=handlers,
            format="%(message)s",
            datefmt=self.timestamp
        )
        return logging.getLogger(__name__)


if __name__ == '__main__':
    my = Config('./my_model.yaml')

