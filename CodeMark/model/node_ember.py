import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class TextualLSTMNodeEmbed(nn.Module):
    def __init__(self, config, codebert=None):
        super(TextualLSTMNodeEmbed, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.word_emb_dims,
                                      padding_idx=self.config.tokenizer.get_pad_id())
        # self.embedding = codebert.roberta.embeddings.word_embeddings

        # with torch.no_grad():
        #     self.embedding.weight.copy_(self.config.clr.transformer.base_model.embeddings.word_embeddings.weight)
        # with torch.no_grad():
        #     self.embedding.weight.copy_(self.config.clr.transformer.base_model.embeddings.word_embeddings.weight)
        # for p in self.embedding.parameters():
        #     p.requires_grad = False

        self.embed_layer = nn.LSTM(self.config.node_encoder_in_dims,
                                   self.config.node_encoder_out_dims,
                                   num_layers=self.config.node_encoder_lstm_layers,
                                   bidirectional=False,
                                   batch_first=True)

    def embed_node(self, node_tok_ids):
        return self.embedding(node_tok_ids)

    def get_var_emb(self, var_logits):
        var_onehots = F.gumbel_softmax(F.log_softmax(var_logits, dim=-1), tau=0.5, hard=True)
        return torch.matmul(var_onehots, self.embedding.weight)

    def get_lstm_output(self, embed_feats, node_lens):
        packed_src = pack_padded_sequence(embed_feats, node_lens, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.embed_layer(packed_src)
        outputs, input_sizes = pad_packed_sequence(outputs, batch_first=True)
        last_seq_ids = torch.tensor([x - 1 for x in node_lens], dtype=torch.long)
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_ids, :]
        return last_seq_items

    def forward(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_lstm_output(embed_feats, node_lens)


class FuncGru(nn.Module):
    def __init__(self, config, embedding):
        super(FuncGru, self).__init__()
        self.config = config

        self.embedding = embedding
        self.gru = nn.GRU(self.config.FuncGru['in_dims'],
                          self.config.FuncGru['out_dims'],
                          num_layers=self.config.FuncGru['n_layers'],
                          batch_first=True)

    def get_gru_output(self, embed_feats, node_lens):
        packed_src = pack_padded_sequence(embed_feats, node_lens, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed_src)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        last_seq_ids = torch.tensor([x - 1 for x in node_lens], dtype=torch.long)
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_ids, :]
        return last_seq_items

    def forward(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_gru_output(embed_feats, node_lens)