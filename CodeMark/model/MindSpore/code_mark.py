import mindspore.nn as nn
from mindspore import ops
from .hiding import Hiding
from configs import Config
from .revealing import Revealing
from .node_ember import TextualLSTMNodeEmbed, FuncGru
from transformers import RobertaForMaskedLM  # Note: May need MindSpore version if available


class CodeMark(nn.Cell):  # Changed from nn.Module to nn.Cell
    def __init__(self, config: Config):
        super(CodeMark, self).__init__()
        self.config = config

        # self.codebert = RobertaForMaskedLM.from_pretrained(config.code_bert_path, local_files_only=True)
        # for p in self.codebert.parameters():
        #     p.requires_grad = False
        # config.logger.info('codemark load codebert')

        # self.node_encoder = TextualLSTMNodeEmbed(config, self.codebert)
        self.node_encoder = TextualLSTMNodeEmbed(config)

        # self.func_gru = FuncGru(config, self.node_encoder.embedding)
        self.func_gru = None

        # self.hiding = Hiding(config, self.node_encoder, self.codebert)
        self.hiding = Hiding(config, self.node_encoder, self.func_gru)
        # self.revealing = Revealing(config, self.node_encoder, self.codebert)
        self.revealing = Revealing(config, self.node_encoder, self.func_gru)

    def construct(self, batch):  # Changed from forward to construct
        rename_var_tok_embs, pre_var_logits, feats = self.hiding.get_rename_pos_and_var(batch)
        pre_watermark_class, pre_feats = self.revealing.get_(batch=batch, rename_var_tok_embs=rename_var_tok_embs)
        return pre_watermark_class, pre_var_logits, feats, pre_feats

    def embed(self, batch):
        # current we only support one function one time
        pre_vars = self.hiding.inference(batch)
        return pre_vars

    def extract(self, batch):
        watermarks = self.revealing.inference(batch)
        return watermarks