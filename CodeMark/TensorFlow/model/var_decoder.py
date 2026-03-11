import tensorflow as tf
from tensorflow.keras import layers

class VarDecoder(tf.keras.Model):
    def __init__(self, config):
        super(VarDecoder, self).__init__()
        self.config = config

        if self.config['VarDecoder']['cat_func_g_emb']:
            self.merge_func_g_emb = layers.Dense(
                self.config['VarDecoder']['cat_out_dims']
            )

        self.lstm = layers.LSTM(
            self.config['VarDecoder']['lstm']['out_dims'],
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.lstm_input_dim = self.config['VarDecoder']['lstm']['in_dims']
        self.linear = layers.Dense(self.config['vocab_size'])

    def call(self, batch, context, training=False):
        max_subtoken_len = max(batch['pre_var_tok_lens'])
        var_num = len(batch['var_tok_lens'])

        var_logits = tf.TensorArray(tf.float32, size=max_subtoken_len)
        hidden = cell = None

        if self.config['VarDecoder']['cat_func_g_emb']:
            context = self.merge_func_g_emb(context)

        for i in range(max_subtoken_len):
            logit, hidden, cell = self.forward_step(context, hidden, cell)
            var_logits = var_logits.write(i, logit)

        var_logits = tf.transpose(var_logits.stack(), perm=[1, 0, 2])  # (batch, seq, vocab)
        return var_logits

    def forward_step(self, x, hidden, cell):
        x = tf.expand_dims(x, axis=0)
        if hidden is None or cell is None:
            output, h, c = self.lstm(x)
        else:
            output, h, c = self.lstm(x, initial_state=[hidden, cell])

        output = tf.squeeze(output, axis=0)
        prediction = self.linear(output)
        return prediction, h, c
class VarDecoderGru(tf.keras.Model):
    def __init__(self, config, embedding_layer):
        super(VarDecoderGru, self).__init__()
        self.config = config
        self.embedding = embedding_layer
        self.gru = layers.GRU(
            self.config['VarDecoder']['gru']['out_dims'],
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            recurrent_dropout=0.0
        )
        self.out = layers.Dense(self.config['vocab_size'])
        self.topk = self.config['VarDecoder']['topk']

    def call(self, batch, context, training=False):
        return self._generate(batch, context, training)

    def _generate(self, batch, context, training):
        batch_size = len(batch['var_tok_lens'])
        max_len = max(batch['pre_var_tok_lens'])

        decoder_input = tf.fill([batch_size, 1], self.config['tokenizer'].bos_token_id)
        decoder_hidden = tf.expand_dims(context, axis=0)

        if training:
            var_logits = []
        else:
            var_ids = []
            generated_hist = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            if self.config['VarDecoder']['substitute_mask_probability'] > 0:
                decoder_output = self._mask_outputs(decoder_output)

            decoder_output = tf.squeeze(decoder_output, axis=1)
            topk_values, topk_indices = tf.math.top_k(decoder_output, k=self.topk)
            random_indices = tf.random.uniform([batch_size, 1], maxval=self.topk, dtype=tf.int32)
            selected_ids = tf.gather(topk_indices, random_indices, batch_dims=1)

            decoder_input = selected_ids

            if training:
                var_logits.append(tf.expand_dims(decoder_output, axis=1))
            else:
                var_ids.append(selected_ids)
                for b in range(batch_size):
                    generated_hist[b].append(int(selected_ids[b]))

        if training:
            return tf.concat(var_logits, axis=1)
        else:
            return tf.concat(var_ids, axis=1)

    def forward_step(self, x, hidden):
        x = self.embedding(x)
        x = tf.nn.leaky_relu(x)
        output, state = self.gru(x, initial_state=hidden)
        output = self.out(output)
        return output, state

    def _mask_outputs(self, outputs):
        k = 10
        topk_values, topk_indices = tf.math.top_k(outputs, k)
        size = tf.shape(topk_indices)
        sample_count = tf.cast(k * self.config['VarDecoder']['substitute_mask_probability'], tf.int32)
        random_indices = tf.random.uniform([size[0], 1], maxval=k, dtype=tf.int32)
        selected = tf.gather(topk_indices, random_indices, batch_dims=1)
        scatter_mask = tf.ones_like(selected, dtype=tf.float32) * 1e-5
        updates = tf.scatter_nd(tf.expand_dims(selected, -1), scatter_mask, tf.shape(outputs))
        return outputs + updates

    def _apply_repetition_penalty(self, logits, generated_sequences):
        for b, tokens in enumerate(generated_sequences):
            for token in set(tokens):
                logits = logits.numpy()
                logits[b, 0, token] = float('-inf')
        return tf.convert_to_tensor(logits)
class VarSelector(tf.keras.Model):
    def __init__(self, config):
        super(VarSelector, self).__init__()
        self.config = config

        self.selector = tf.keras.Sequential([
            layers.Dense(self.config['VarSelector']['hidden_dim']),
            layers.LeakyReLU(),
            layers.Dense(2)
        ])

    def call(self, x, training=False):
        x = self.selector(x)
        if self.config['VarSelector']['use_mask']:
            x = self._mask_pre(x)
        return x

    def _mask_pre(self, x):
        batch_size = tf.shape(x)[0]
        pre_mask = tf.random.uniform((batch_size, 1))
        ori_mask = tf.zeros((batch_size, 1))
        mask = tf.concat([pre_mask, ori_mask], axis=1)
        mask = mask > self.config['VarSelector']['mask_probability']
        x = tf.where(mask, tf.constant(float('-inf'), dtype=x.dtype), x)
        return x
