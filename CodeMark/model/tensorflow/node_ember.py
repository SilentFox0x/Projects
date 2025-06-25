import tensorflow as tf

class TextualLSTMNodeEmbed(tf.keras.Model):
    def __init__(self, config, codebert_embedding=None):
        super(TextualLSTMNodeEmbed, self).__init__()
        self.config = config

        if codebert_embedding is not None:
            self.embedding = codebert_embedding
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=self.config['vocab_size'],
                output_dim=self.config['word_emb_dims'],
                mask_zero=True  # 用于 padding
            )

        self.lstm = tf.keras.layers.LSTM(
            units=self.config['node_encoder_out_dims'],
            return_sequences=True,
            return_state=True
        )

    def embed_node(self, node_tok_ids):
        return self.embedding(node_tok_ids)

    def get_var_emb(self, var_logits):
        # Gumbel softmax + embedding lookup
        var_onehots = tf.nn.gumbel_softmax(tf.nn.log_softmax(var_logits, axis=-1), tau=0.5, hard=True)
        return tf.matmul(var_onehots, self.embedding.weights[0])

    def get_lstm_output(self, embed_feats, node_lens):
        mask = tf.sequence_mask(node_lens)
        outputs, h, c = self.lstm(embed_feats, mask=mask)
        last_indices = tf.expand_dims(node_lens - 1, axis=1)
        batch_indices = tf.range(tf.shape(node_lens)[0])[:, tf.newaxis]
        gather_indices = tf.concat([batch_indices, last_indices], axis=1)
        last_seq_items = tf.gather_nd(outputs, gather_indices)
        return last_seq_items

    def call(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_lstm_output(embed_feats, node_lens)
class FuncGru(tf.keras.Model):
    def __init__(self, config, embedding_layer):
        super(FuncGru, self).__init__()
        self.config = config
        self.embedding = embedding_layer
        self.gru = tf.keras.layers.GRU(
            units=self.config['FuncGru']['out_dims'],
            return_sequences=True,
            return_state=True
        )

    def get_gru_output(self, embed_feats, node_lens):
        mask = tf.sequence_mask(node_lens)
        outputs, hidden = self.gru(embed_feats, mask=mask)
        last_indices = tf.expand_dims(node_lens - 1, axis=1)
        batch_indices = tf.range(tf.shape(node_lens)[0])[:, tf.newaxis]
        gather_indices = tf.concat([batch_indices, last_indices], axis=1)
        last_seq_items = tf.gather_nd(outputs, gather_indices)
        return last_seq_items

    def call(self, node_tok_ids, node_lens):
        embed_feats = self.embedding(node_tok_ids)
        return self.get_gru_output(embed_feats, node_lens)
