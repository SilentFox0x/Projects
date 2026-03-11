import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class TextualLSTMNodeEmbed(nn.Cell):
    def __init__(self, config, codebert=None):
        super(TextualLSTMNodeEmbed, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.word_emb_dims,
                                      padding_idx=self.config.tokenizer.get_pad_id())

        self.embed_layer = nn.LSTM(self.config.node_encoder_in_dims,
                                   self.config.node_encoder_out_dims,
                                   num_layers=self.config.node_encoder_lstm_layers,
                                   batch_first=True)

    def embed_node(self, node_tok_ids):
        return self.embedding(node_tok_ids)

    def get_var_emb(self, var_logits):
        log_softmax = ops.LogSoftmax(axis=-1)
        var_onehots = ops.gumbel_softmax(log_softmax(var_logits), tau=0.5, hard=True)
        return ops.matmul(var_onehots, self.embedding.embedding_table)

    def get_lstm_output(self, embed_feats, node_lens):
        outputs, (hidden, cell) = self.embed_layer(embed_feats)

        batch_indices = ops.Range()(0, embed_feats.shape[0], 1)
        last_seq_ids = Tensor([x - 1 for x in node_lens], mindspore.int32)
        idx = ops.Stack(1)((batch_indices, last_seq_ids))
        last_seq_items = ops.GatherNd()(outputs, idx)
        return last_seq_items

    def construct(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_lstm_output(embed_feats, node_lens)
class FuncGru(nn.Cell):
    def __init__(self, config, embedding):
        super(FuncGru, self).__init__()
        self.config = config
        self.embedding = embedding

        self.gru = nn.GRU(self.config.FuncGru['in_dims'],
                          self.config.FuncGru['out_dims'],
                          num_layers=self.config.FuncGru['n_layers'],
                          batch_first=True)

    def get_gru_output(self, embed_feats, node_lens):
        outputs, hidden = self.gru(embed_feats)

        batch_indices = ops.Range()(0, embed_feats.shape[0], 1)
        last_seq_ids = Tensor([x - 1 for x in node_lens], mindspore.int32)
        idx = ops.Stack(1)((batch_indices, last_seq_ids))
        last_seq_items = ops.GatherNd()(outputs, idx)
        return last_seq_items

    def construct(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_gru_output(embed_feats, node_lens)
