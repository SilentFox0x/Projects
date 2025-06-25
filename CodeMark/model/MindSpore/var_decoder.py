import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class VarDecoder(nn.Cell):
    def __init__(self, config):
        super(VarDecoder, self).__init__()
        self.config = config
        if self.config.VarDecoder['cat_func_g_emb']:
            self.merge_func_g_emb = nn.Dense(self.config.VarDecoder['cat_in_dims'],
                                             self.config.VarDecoder['cat_out_dims'])

        self.lstm = nn.LSTM(self.config.VarDecoder['lstm']['in_dims'],
                            self.config.VarDecoder['lstm']['out_dims'],
                            num_layers=self.config.VarDecoder['lstm']['n_layers'],
                            bidirectional=False)

        self.linear = nn.Dense(self.config.VarDecoder['lstm']['out_dims'], self.config.vocab_size)

    def forward_step(self, x, hidden, cell):
        x = ops.ExpandDims()(x, 0)  # shape: [1, batch, feature]

        if hidden is None and cell is None:
            output, (hidden, cell) = self.lstm(x)
        else:
            output, (hidden, cell) = self.lstm(x, (hidden, cell))

        output = ops.Squeeze(0)(output)
        prediction = self.linear(output)
        return prediction, hidden, cell

    def construct(self, batch, context):
        max_subtoken_len = max(batch['pre_var_tok_lens'])
        var_num = len(batch['var_tok_lens'])
        var_logits = ops.Zeros()((max_subtoken_len, var_num, self.config.vocab_size), mindspore.float32)
        var_logits = var_logits.to(self.config.device)
        hidden, cell = None, None

        if self.config.VarDecoder['cat_func_g_emb']:
            context = self.merge_func_g_emb(context)

        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.forward_step(context, hidden, cell)
            var_logits[i] = var_logit

        var_logits = ops.Transpose()(var_logits, (1, 0, 2))
        return var_logits
class VarDecoderGru(nn.Cell):
    def __init__(self, config, embedding):
        super(VarDecoderGru, self).__init__()
        self.config = config
        self.embedding = embedding
        self.gru = nn.GRU(self.config.VarDecoder['gru']['in_dims'],
                          self.config.VarDecoder['gru']['out_dims'],
                          num_layers=self.config.VarDecoder['gru']['n_layers'],
                          batch_first=True)
        self.out = nn.Dense(self.config.VarDecoder['gru']['out_dims'], self.config.vocab_size)
        self.topk = self.config.VarDecoder['topk']

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = ops.LeakyRelu()(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def _mask_outputs(self, outputs):
        k = 10
        topk = ops.TopK(sorted=True)
        _, top_indices = topk(outputs, k)

        size = list(top_indices.shape)
        size[-1] = int(k * self.config.VarDecoder['substitute_mask_probability'])
        random_indices = Tensor(np.random.randint(0, k, size), mindspore.int32)
        selected_values = ops.GatherD()(top_indices, 2, random_indices)

        scatter = ops.ScatterNdUpdate()
        outputs = scatter(outputs, selected_values, Tensor(1e-5, mindspore.float32))
        return outputs

    def construct(self, batch, context):
        return self._generate(batch, context, is_training=True)

    def inference(self, batch, context):
        return self._generate(batch, context, is_training=False)

    def _generate(self, batch, context, is_training):
        batch_size = len(batch['var_tok_lens'])
        decoder_input = ops.Fill()(mindspore.int64, (batch_size, 1), self.config.tokenizer.bos_token_id)
        decoder_hidden = ops.ExpandDims()(context, 0)

        if is_training:
            var_logits = []
        else:
            var_ids = []
            generated_hist = [[] for _ in range(batch_size)]

        max_subtoken_len = max(batch['pre_var_tok_lens'])
        for i in range(max_subtoken_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            if self.config.VarDecoder['substitute_mask_probability'] > 0:
                decoder_output = self._mask_outputs(decoder_output)

            if is_training:
                var_logits.append(decoder_output)
            else:
                decoder_output = self._apply_repetition_penalty(decoder_output, generated_hist)

            decoder_output = ops.Squeeze(1)(decoder_output)
            topk = ops.TopK()
            _, topi = topk(decoder_output, self.topk)

            random_indices = Tensor(np.random.randint(0, self.topk, (batch_size, 1)), mindspore.int32)
            decoder_input = ops.GatherD()(topi, 1, random_indices)

            if not is_training:
                var_ids.append(decoder_input)
                for b in range(batch_size):
                    generated_hist[b].append(int(decoder_input[b]))

        if is_training:
            return ops.Concat(1)(var_logits)
        else:
            return ops.Concat(1)(var_ids)

    def _apply_repetition_penalty(self, logits, generated_sequences):
        for b in range(len(generated_sequences)):
            for token in set(generated_sequences[b]):
                logits[b, 0, token] = float('-inf')
        return logits
class VarSelector(nn.Cell):
    def __init__(self, config):
        super(VarSelector, self).__init__()
        self.config = config

        self.selector = nn.SequentialCell(
            nn.Dense(self.config.VarSelector['input_dim'], self.config.VarSelector['hidden_dim']),
            nn.LeakyReLU(),
            nn.Dense(self.config.VarSelector['hidden_dim'], 2),
        )

    def _mask_pre(self, x):
        pre_mask = ops.UniformReal()((x.shape[0], 1))
        ori_mask = ops.Zeros()((x.shape[0], 1), mindspore.float32)
        mask = ops.Concat(1)((pre_mask, ori_mask)) > self.config.VarSelector['mask_probability']
        x = ops.MaskedFill()(x, mask, float('-inf'))
        return x

    def construct(self, x):
        x = self.selector(x)
        if self.config.VarSelector['use_mask']:
            x = self._mask_pre(x)
        return x
