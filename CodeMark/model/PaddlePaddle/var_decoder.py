import paddle


class VarDecoder(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.VarDecoder["cat_func_g_emb"]:
            self.merge_func_g_emb = paddle.nn.Linear(
                in_features=self.config.VarDecoder["cat_in_dims"],
                out_features=self.config.VarDecoder["cat_out_dims"],
            )
        self.lstm = paddle.nn.LSTM(
            input_size=self.config.VarDecoder["lstm"]["in_dims"],
            hidden_size=self.config.VarDecoder["lstm"]["out_dims"],
            num_layers=self.config.VarDecoder["lstm"]["n_layers"],
            time_major=not False,
        )
        self.linear = paddle.nn.Linear(
            in_features=self.config.VarDecoder["lstm"]["out_dims"],
            out_features=self.config.vocab_size,
        )

    def forward_step(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        x = x.unsqueeze(axis=0)
        if hidden is None and cell is None:
            output, (hidden, cell) = self.lstm(x)
        else:
            output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = output.squeeze(axis=0)
        prediction = self.linear(output)
        return prediction, hidden, cell

    def forward(self, batch, context):
        max_subtoken_len = max(batch["pre_var_tok_lens"])
        var_num = len(batch["var_tok_lens"])
        var_logits = paddle.zeros(
            shape=[max_subtoken_len, var_num, self.config.vocab_size]
        ).to(self.config.place)
        hidden, cell = None, None
        if self.config.VarDecoder["cat_func_g_emb"]:
            context = self.merge_func_g_emb(context)
        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.forward_step(context, hidden, cell)
            var_logits[i] = var_logit
        var_logits = paddle.transpose(x=var_logits, perm=(1, 0, 2))
        return var_logits


class VarDecoderGru(paddle.nn.Layer):
    def __init__(self, config, embedding):
        super(VarDecoderGru, self).__init__()
        self.config = config
        self.embedding = embedding
        self.gru = paddle.nn.GRU(
            input_size=self.config.VarDecoder["gru"]["in_dims"],
            hidden_size=self.config.VarDecoder["gru"]["out_dims"],
            num_layers=self.config.VarDecoder["gru"]["n_layers"],
            time_major=not True,
        )
        self.out = paddle.nn.Linear(
            in_features=self.config.VarDecoder["gru"]["out_dims"],
            out_features=self.config.vocab_size,
        )
        self.topk = self.config.VarDecoder["topk"]

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = paddle.nn.functional.leaky_relu(x=output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def _mask_outputs(self, outputs):
        k = 10
        _, top_indices = paddle.topk(x=outputs, k=k)
        size = list(tuple(top_indices.shape))
        size[-1] = int(k * self.config.VarDecoder["substitute_mask_probability"])
        random_indices = paddle.randint(low=0, high=k, shape=size).to(self.config.place)
        selected_values = paddle.take_along_axis(
            arr=top_indices, axis=2, indices=random_indices, broadcast=False
        )
        outputs.put_along_axis_(
            axis=2, indices=selected_values, values=1e-05, broadcast=False
        )
        return outputs

    def forward(self, batch, context):
        return self._generate(batch, context, is_training=True)

    def inference(self, batch, context):
        return self._generate(batch, context, is_training=False)

    def _generate(self, batch, context, is_training):
        batch_size = len(batch["var_tok_lens"])
        decoder_input = paddle.empty(shape=[batch_size, 1], dtype="int64").fill_(
            value=self.config.tokenizer.bos_token_id
        )
        decoder_hidden = paddle.unsqueeze(x=context, axis=0)
        if is_training:
            var_logits = []
        else:
            var_ids = []
            generated_hist = [[] for _ in range(batch_size)]
        max_subtoken_len = max(batch["pre_var_tok_lens"])
        for i in range(max_subtoken_len):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            if self.config.VarDecoder["substitute_mask_probability"] > 0:
                decoder_output = self._mask_outputs(decoder_output)
            if is_training:
                var_logits.append(decoder_output)
            else:
                decoder_output = self._apply_repetition_penalty(
                    decoder_output, generated_hist
                )
            decoder_output = decoder_output.squeeze(axis=1)
            _, topi = decoder_output.topk(k=self.topk)
            random_indices = paddle.randint(
                low=0, high=self.topk, shape=(batch_size, 1)
            )
            topi = topi.take_along_axis(axis=1, indices=random_indices, broadcast=False)
            decoder_input = topi.detach()
            if not is_training:
                var_ids.append(topi)
                for b in range(batch_size):
                    generated_hist[b].append(topi[b].item())
        if is_training:
            var_logits = paddle.concat(x=var_logits, axis=1)
            return var_logits
        else:
            var_ids = paddle.concat(x=var_ids, axis=1)
            return var_ids

    def _apply_repetition_penalty(self, logits, generated_sequences):
        """
        logits: (batch_size, 1, vocab_size)
        generated_sequences: 列表的列表，每个内部列表表示对应样本的生成历史
        """
        batch_size = logits.shape[0]
        for b in range(batch_size):
            recent_tokens = generated_sequences[b]
            for token in set(recent_tokens):
                logits[b, 0, token] = float("-inf")
        return logits


class VarSelector(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.selector = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=self.config.VarSelector["input_dim"],
                out_features=self.config.VarSelector["hidden_dim"],
            ),
            paddle.nn.LeakyReLU(),
            paddle.nn.Linear(
                in_features=self.config.VarSelector["hidden_dim"], out_features=2
            ),
        )

    def _mask_pre(self, x):
        pre_mask = paddle.rand(shape=(tuple(x.shape)[0], 1))
        ori_mask = paddle.zeros(shape=(tuple(x.shape)[0], 1))
        mask = paddle.concat(x=[pre_mask, ori_mask], axis=1).to(self.config.place)
        mask = mask > self.config.VarSelector["mask_probability"]
        x = paddle.masked_fill(x=x, mask=mask.astype(dtype="bool"), value=float("-inf"))
        return x

    def forward(self, x):
        x = self.selector(x)
        if self.config.VarSelector["use_mask"]:
            x = self._mask_pre(x)
        return x
