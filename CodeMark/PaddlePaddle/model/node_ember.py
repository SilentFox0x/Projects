import sys

sys.path.append("/home/liwei/paddlepaddle")
import paddle
from paddle_utils import *


class TextualLSTMNodeEmbed(paddle.nn.Layer):
    def __init__(self, config, codebert=None):
        super(TextualLSTMNodeEmbed, self).__init__()
        self.config = config
        self.embedding = Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.word_emb_dims,
            padding_idx=self.config.tokenizer.get_pad_id(),
        )
        self.embed_layer = paddle.nn.LSTM(
            input_size=self.config.node_encoder_in_dims,
            hidden_size=self.config.node_encoder_out_dims,
            num_layers=self.config.node_encoder_lstm_layers,
            time_major=not True,
        )

    def embed_node(self, node_tok_ids):
        return self.embedding(node_tok_ids)

    def get_var_emb(self, var_logits):
        var_onehots = paddle.nn.functional.gumbel_softmax(
            x=paddle.nn.functional.log_softmax(x=var_logits, axis=-1),
            temperature=0.5,
            hard=True,
        )
        return paddle.matmul(x=var_onehots, y=self.embedding.weight)

    def get_lstm_output(self, embed_feats, node_lens):
>>>>>>        packed_src = torch.nn.utils.rnn.pack_padded_sequence(
            embed_feats, node_lens, batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.embed_layer(packed_src)
>>>>>>        outputs, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )
        last_seq_ids = paddle.to_tensor(
            data=[(x - 1) for x in node_lens], dtype="int64"
        )
        last_seq_items = outputs[range(tuple(outputs.shape)[0]), last_seq_ids, :]
        return last_seq_items

    def forward(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_lstm_output(embed_feats, node_lens)


class FuncGru(paddle.nn.Layer):
    def __init__(self, config, embedding):
        super(FuncGru, self).__init__()
        self.config = config
        self.embedding = embedding
        self.gru = paddle.nn.GRU(
            input_size=self.config.FuncGru["in_dims"],
            hidden_size=self.config.FuncGru["out_dims"],
            num_layers=self.config.FuncGru["n_layers"],
            time_major=not True,
        )

    def get_gru_output(self, embed_feats, node_lens):
>>>>>>        packed_src = torch.nn.utils.rnn.pack_padded_sequence(
            embed_feats, node_lens, batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.gru(packed_src)
>>>>>>        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        last_seq_ids = paddle.to_tensor(
            data=[(x - 1) for x in node_lens], dtype="int64"
        )
        last_seq_items = outputs[range(tuple(outputs.shape)[0]), last_seq_ids, :]
        return last_seq_items

    def forward(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_gru_output(embed_feats, node_lens)
