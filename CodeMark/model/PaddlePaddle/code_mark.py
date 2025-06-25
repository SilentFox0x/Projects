import paddle
import paddlenlp
from configs import Config

from .hiding import Hiding
from .node_ember import FuncGru, TextualLSTMNodeEmbed
from .revealing import Revealing


class CodeMark(paddle.nn.Layer):
    def __init__(self, config: Config):
        super(CodeMark, self).__init__()
        self.config = config
        self.node_encoder = TextualLSTMNodeEmbed(config)
        self.func_gru = None
        self.hiding = Hiding(config, self.node_encoder, self.func_gru)
        self.revealing = Revealing(config, self.node_encoder, self.func_gru)

    def forward(self, batch):
        rename_var_tok_embs, pre_var_logits, feats = self.hiding.get_rename_pos_and_var(
            batch
        )
        pre_watermark_class, pre_feats = self.revealing.get_(
            batch=batch, rename_var_tok_embs=rename_var_tok_embs
        )
        return pre_watermark_class, pre_var_logits, feats, pre_feats

    def embed(self, batch):
        pre_vars = self.hiding.inference(batch)
        return pre_vars

    def extract(self, batch):
        watermarks = self.revealing.inference(batch)
        return watermarks
