import tensorflow as tf
from .hiding import HidingTF
from configs import Config
from .revealing import RevealingTF
from .node_ember import TextualLSTMNodeEmbed

class CodeMarkTF(tf.keras.Model):
    def __init__(self, config):
        super(CodeMarkTF, self).__init__()
        self.config = config

        # 1. 节点编码器
        self.node_encoder = TextualLSTMNodeEmbed(config)

        # 2. 可选：函数级 GRU（当前为 None）
        self.func_gru = None  # or: FuncGru(config, self.node_encoder.embedding)

        # 3. 水印嵌入模型
        self.hiding = HidingTF(config, self.node_encoder, self.func_gru)

        # 4. 水印提取模型
        self.revealing = RevealingTF(config, self.node_encoder, self.func_gru)

    def call(self, batch, training=False):
        # 正向传播：嵌入 + 提取 + 监督
        rename_var_tok_embs, pre_var_logits, feats = self.hiding.get_rename_pos_and_var(batch)
        pre_watermark_class, pre_feats = self.revealing.get_(batch=batch, rename_var_tok_embs=rename_var_tok_embs)
        return pre_watermark_class, pre_var_logits, feats, pre_feats

    def embed(self, batch):
        # 仅执行水印嵌入（变量名生成）
        return self.hiding.inference(batch)

    def extract(self, batch):
        # 仅执行水印提取
        return self.revealing.inference(batch)
