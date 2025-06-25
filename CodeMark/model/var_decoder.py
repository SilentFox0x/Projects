import torch.nn as nn
import torch.nn.functional as F
import torch


class VarDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.VarDecoder['cat_func_g_emb']:
            self.merge_func_g_emb = nn.Linear(self.config.VarDecoder['cat_in_dims'],
                                              self.config.VarDecoder['cat_out_dims'])

        self.lstm = nn.LSTM(self.config.VarDecoder['lstm']['in_dims'],
                            self.config.VarDecoder['lstm']['out_dims'],
                            num_layers=self.config.VarDecoder['lstm']['n_layers'],
                            bidirectional=False)

        self.linear = nn.Linear(self.config.VarDecoder['lstm']['out_dims'], self.config.vocab_size)

    def forward_step(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(0)

        # embedded = [1, batch size, embedding size]

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hidden size]
        # hidden = [n layers, batch size, hidden size]
        # cell = [n layers, batch size, hidden size]
        if hidden is None and cell is None:
            output, (hidden, cell) = self.lstm(x)
        else:
            output, (hidden, cell) = self.lstm(x, (hidden, cell))

        output = output.squeeze(0)
        # prediction = [batch size, output size]
        prediction = self.linear(output)
        # pre_var_vocab_hot = F.gumbel_softmax(F.log_softmax(prediction, dim=-1), tau=0.5, hard=True)
        return prediction, hidden, cell

    def forward(self, batch, context):
        max_subtoken_len = max(batch['pre_var_tok_lens'])
        var_num = len(batch['var_tok_lens'])
        var_logits = torch.zeros(max_subtoken_len, var_num, self.config.vocab_size).to(self.config.device)
        hidden, cell = None, None

        if self.config.VarDecoder['cat_func_g_emb']:
            context = self.merge_func_g_emb(context)

        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.forward_step(context, hidden, cell)
            var_logits[i] = var_logit
        # (subtoken_len, batch, vocab) --> (batch, subtoken_len, vocab)
        var_logits = torch.permute(var_logits, (1, 0, 2))
        return var_logits


class VarDecoderGru(nn.Module):
    def __init__(self, config, embedding):
        super(VarDecoderGru, self).__init__()
        self.config = config
        self.embedding = embedding
        self.gru = nn.GRU(self.config.VarDecoder['gru']['in_dims'],
                          self.config.VarDecoder['gru']['out_dims'],
                          num_layers=self.config.VarDecoder['gru']['n_layers'],
                          batch_first=True)
        self.out = nn.Linear(self.config.VarDecoder['gru']['out_dims'], self.config.vocab_size)
        self.topk = self.config.VarDecoder['topk']

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.leaky_relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def _mask_outputs(self, outputs):
        k = 10
        _, top_indices = torch.topk(outputs, k)

        # 从top-k中，随机将一些值置低
        size = list(top_indices.shape)
        size[-1] = int(k * self.config.VarDecoder['substitute_mask_probability'])
        random_indices = torch.randint(0, k, size=size).to(self.config.device)
        selected_values = torch.gather(top_indices, dim=2, index=random_indices)

        outputs.scatter_(2, selected_values, 1e-5)
        return outputs

    def forward(self, batch, context):
        return self._generate(batch, context, is_training=True)

    def inference(self, batch, context):
        return self._generate(batch, context, is_training=False)

    def _generate(self, batch, context, is_training):
        batch_size = len(batch['var_tok_lens'])
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.config.device).fill_(
            self.config.tokenizer.bos_token_id)
        decoder_hidden = torch.unsqueeze(context, dim=0)

        if is_training:
            var_logits = []
        else:
            var_ids = []
            generated_hist = [[] for _ in range(batch_size)]  # 二维列表[batch][time_step]

        max_subtoken_len = max(batch['pre_var_tok_lens'])
        for i in range(max_subtoken_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            if self.config.VarDecoder['substitute_mask_probability'] > 0:
                decoder_output = self._mask_outputs(decoder_output)

            if is_training:
                var_logits.append(decoder_output)
            else:
                decoder_output = self._apply_repetition_penalty(decoder_output, generated_hist)

            # 在一般的NLP任务重，训练时应该用teacher force，但是水印任务没有，所以需要自己构造
            # 如果直接用前一个token，很容易造成repeat token
            # 所以我们每次随机从 top-10 中选择一个 token
            decoder_output = decoder_output.squeeze(1)
            _, topi = decoder_output.topk(self.topk)
            random_indices = torch.randint(0, self.topk, (batch_size, 1), device=self.config.device)
            topi = topi.gather(1, random_indices)
            decoder_input = topi.detach()

            if not is_training:
                var_ids.append(topi)
                for b in range(batch_size):
                    generated_hist[b].append(topi[b].item())

        if is_training:
            var_logits = torch.cat(var_logits, dim=1)
            return var_logits
        else:
            var_ids = torch.cat(var_ids, dim=1)
            return var_ids

    def _apply_repetition_penalty(self, logits, generated_sequences):
        """
        logits: (batch_size, 1, vocab_size)
        generated_sequences: 列表的列表，每个内部列表表示对应样本的生成历史
        """
        batch_size = logits.size(0)
        for b in range(batch_size):
            recent_tokens = generated_sequences[b]
            # 对重复token进行去重后惩罚
            for token in set(recent_tokens):
                logits[b, 0, token] = float('-inf')
        return logits


class VarSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.selector = nn.Sequential(
            nn.Linear(self.config.VarSelector['input_dim'], self.config.VarSelector['hidden_dim']),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.config.VarSelector['hidden_dim'], 2),
        )

    def _mask_pre(self, x):
        pre_mask = torch.rand((x.shape[0], 1))
        ori_mask = torch.zeros((x.shape[0], 1))  # not mask
        mask = torch.cat([pre_mask, ori_mask], dim=1).to(self.config.device)
        mask = mask > self.config.VarSelector['mask_probability']
        x = torch.masked_fill(x, mask.bool(), float('-inf'))
        return x

    def forward(self, x):
        x = self.selector(x)
        # if self.config.VarSelector['use_mask'] and self.config.mode != 'test':
        if self.config.VarSelector['use_mask']:
            x = self._mask_pre(x)
        return x
