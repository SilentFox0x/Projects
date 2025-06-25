import tensorflow as tf
from .gat import MyGatModel

class RevealingTF(tf.keras.Model):
    def __init__(self, config, node_encoder, func_gru=None):
        super(RevealingTF, self).__init__()
        self.config = config
        self.node_encoder = node_encoder
        self.func_gru = func_gru

        self.graph_encoder = MyGatModel(
            layer_num=config['hiding_layers'],
            first_in_feats=config['revealing']['first_in_dim'],
            first_out_feats=config['revealing']['first_out_dim'],
            first_heads=config['revealing']['first_heads'],
            second_out_dim=config['revealing']['second_out_dim'],
            second_heads=config['revealing']['second_heads']
        )

        watermark_classes = 1 << config['watermark_len']
        self.watermark_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(config['watermark_decoder_hidden_dims']),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(watermark_classes)
        ])

    def encode_graph(self, var_position_in_node_batch, A, X, pre_var_emb, var_node_index_in_node_batch, pre_var_tok_lens):
        h = self.node_encoder.embed_node(X['node_tok_ids'])

        for node_index, var_hels in enumerate(var_position_in_node_batch):
            for vrh in var_hels:
                pre_len = pre_var_tok_lens[vrh['var_index_in_vars']]
                if vrh['begin'] + pre_len > tf.shape(h)[1]:
                    continue
                h[node_index, vrh['begin']:vrh['begin'] + pre_len, :].assign(
                    pre_var_emb[vrh['var_index_in_vars'], :pre_len, :]
                )

        node_lens = X['node_tok_lens']
        h = self.node_encoder.get_lstm_output(h, node_lens)
        g = self.graph_encoder([X, A])
        g = tf.gather(g, var_node_index_in_node_batch)
        return g

    def call(self, var_position_in_node_batch, graph_inputs, pre_var_logits, var_node_index_in_node_batch, pre_var_tok_lens, watermarks_class):
        A, X = graph_inputs
        pre_var_emb = self.node_encoder.get_var_emb(pre_var_logits)
        g_emb = self.encode_graph(var_position_in_node_batch, A, X, pre_var_emb, var_node_index_in_node_batch, pre_var_tok_lens)
        logits = self.watermark_decoder(g_emb)
        return logits

    def get_func_embs_after_rename(self, func_token_ids, function_token_lens, var_position_in_func_batch, pre_var_emb, pre_var_tok_lens):
        h = self.node_encoder.embed_node(func_token_ids)
        for func_index, var_hels in enumerate(var_position_in_func_batch):
            for vrh in var_hels:
                pre_len = pre_var_tok_lens[vrh['var_index_in_vars']]
                h[func_index, vrh['begin']:vrh['begin'] + pre_len, :].assign(
                    pre_var_emb[vrh['var_index_in_vars'], :pre_len, :]
                )
        return self.func_gru.get_gru_output(h, function_token_lens)

    def get_(self, batch, rename_var_tok_embs):
        h = self.encode_graph(
            var_position_in_node_batch=batch['var_position_in_node_batch'],
            A=batch['graph_batch'][0],
            X=batch['graph_batch'][1],
            pre_var_emb=rename_var_tok_embs,
            var_node_index_in_node_batch=batch['var_node_index_in_node_batch'],
            pre_var_tok_lens=batch['pre_var_tok_lens']
        )

        g_embs = tf.reduce_mean(h, axis=1)

        # max pooling over function → watermark embedding
        pre_wm_embs = []
        for g_begin, g_end in batch['func_map_var_position']:
            pre_wm_embs.append(tf.reduce_max(g_embs[g_begin:g_end], axis=0))
        pre_wm_embs = tf.stack(pre_wm_embs)

        pre_watermark_class = self.watermark_decoder(pre_wm_embs)

        pre_feats = g_embs if self.config['use_distill_and_mse_loss'] else None
        return pre_watermark_class, pre_feats

    def inference(self, batch):
        A, X = batch['graph_batch']
        h = self.node_encoder(X['node_tok_ids'], X['node_tok_lens'])
        g = self.graph_encoder([X, A])
        g = tf.gather(g, batch['var_node_index_in_node_batch'])
        g_embs = tf.reduce_mean(g, axis=1)

        pre_wm_embs = []
        for g_begin, g_end in batch['func_map_var_position']:
            pre_wm_embs.append(tf.reduce_max(g_embs[g_begin:g_end], axis=0))
        pre_wm_embs = tf.stack(pre_wm_embs)

        pre_watermark_class = self.watermark_decoder(pre_wm_embs)

        # 这里你有 utils.get_watermarks_from_w_class 函数，需要你一起提供或补充我可以帮你重写成 TF 版本
        return pre_watermark_class  # or get_watermarks_from_w_class(pre_watermark_class)
