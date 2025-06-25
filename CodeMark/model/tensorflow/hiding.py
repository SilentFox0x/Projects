import tensorflow as tf
from .var_decoder import VarDecoder, VarSelector, VarDecoderGru
from .gat import MyGatModel
from .wm_encoder import WMLinearEncoder


class HidingTF(tf.keras.Model):
    def __init__(self, config, node_encoder, func_gru=None):
        super(HidingTF, self).__init__()
        self.config = config
        self.node_encoder = node_encoder
        self.func_gru = func_gru

        self.wm_encoder = WMLinearEncoder(config['watermark_len'], config['watermark_emb_dims'])

        second_heads = 2 ** config['watermark_len']
        self.graph_encoder = MyGatModel(
            layer_num=config['hiding_layers'],
            first_in_feats=config['hiding']['first_in_dim'],
            first_out_feats=config['hiding']['first_out_dim'],
            first_heads=config['hiding']['first_heads'],
            second_out_dim=config['hiding']['second_out_dim'],
            second_heads=second_heads
        )

        decoder_type = config['VarDecoder']['decoder_type']
        if decoder_type == 'lstm':
            self.var_decoder = VarDecoder(config)
        elif decoder_type == 'gru':
            self.var_decoder = VarDecoderGru(config, node_encoder.embedding)
        else:
            raise TypeError('wrong decoder type')

        self.var_selector = VarSelector(config)

    def encode_graph(self, A, X, var_node_indices, wm_classes):
        h = self.node_encoder(X['node_tok_ids'], X['node_tok_lens'])
        h = self.graph_encoder([X, A])  # (N, H*out)
        h = tf.gather(h, var_node_indices)  # (V, H*out)

        selected = []
        for i, cls in enumerate(wm_classes):
            selected.append(h[i, cls, :])
        return tf.stack(selected, axis=0)

    def call(self, watermarks_class, graph_inputs, var_node_indices, pre_var_tok_lens,
             var_tok_ids, var_tok_lens, training=False):
        g_emb = self.encode_graph(*graph_inputs, var_node_indices, watermarks_class)

        max_len = max(pre_var_tok_lens)
        batch_size = tf.shape(watermarks_class)[0]
        var_logits = tf.TensorArray(tf.float32, size=max_len)

        hidden = cell = None
        for i in range(max_len):
            var_logit, hidden, cell = self.var_decoder.forward_step(g_emb, hidden, cell)
            var_logits = var_logits.write(i, var_logit)

        var_logits = tf.transpose(var_logits.stack(), perm=[1, 0, 2])
        return var_logits
