from typing import Dict

import torch
import torch.nn as nn
from .var_decoder import VarDecoder, VarSelector, VarDecoderGru
from .gat import MyGatModel
from .wm_encoder import WMLinearEncoder
import torch.nn.functional as F
from utils import get_var_map, get_var_from_var_id


class Hiding(nn.Module):
    def __init__(self, config, node_encoder, func_gru=None):
        super(Hiding, self).__init__()
        self.config = config
        self.wm_encoder = WMLinearEncoder(config.watermark_len, config.watermark_emb_dims)

        # self.codebert = codebert

        self.node_encoder = node_encoder
        self.func_gru = func_gru

        second_heads = 2 ** config.watermark_len
        self.graph_encoder = MyGatModel(layer_num=self.config.hiding_layers,
                                        first_in_feats=self.config.hiding['first_in_dim'],
                                        first_out_feats=self.config.hiding['first_out_dim'],
                                        first_heads=self.config.hiding['first_heads'],
                                        second_out_dim=self.config.hiding['second_out_dim'],
                                        second_heads=second_heads)

        if self.config.VarDecoder['decoder_type'] == 'lstm':
            self.var_decoder = VarDecoder(config)
        elif self.config.VarDecoder['decoder_type'] == 'gru':
            self.var_decoder = VarDecoderGru(config, node_encoder.embedding)
        else:
            raise TypeError('wrong decoder type')
        self.var_selector = VarSelector(config)

    def encode_graph(self, graphs, var_node_index_in_node_batch, watermarks_class):
        h = graphs.ndata['node_tok_ids']
        node_len = graphs.ndata['node_tok_lens'].cpu().tolist()
        h = self.node_encoder(h, node_len)

        h = self.graph_encoder(graphs, h)  # [node_size, nhead, dims]
        h = torch.index_select(h, 0, var_node_index_in_node_batch)
        h = [h[index, w_class, :] for index, w_class in enumerate(watermarks_class)]
        h = torch.stack(h)
        # h = torch.squeeze(h)

        return h

    def forward(self, watermarks_class, graph_batch, var_node_index_in_node_batch, pre_var_tok_lens,
                var_tok_ids, var_tok_lens):
        g_emb = self.encode_graph(graph_batch, var_node_index_in_node_batch, watermarks_class)

        max_subtoken_len = max(pre_var_tok_lens)
        batch_size = watermarks_class.shape[0]
        var_logits = torch.zeros(max_subtoken_len, batch_size, self.config.vocab_size).to(self.config.device)
        hidden, cell = None, None
        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.var_decoder(g_emb, hidden, cell)
            var_logits[i] = var_logit

        # (subtoken_len, batch, vocab) --> (batch, subtoken_len, vocab)
        var_logits = torch.permute(var_logits, (1, 0, 2))
        return var_logits

    def get_g_embs(self, graph_batch, var_node_index_in_node_batch):
        h = graph_batch.ndata['node_tok_ids']
        node_len = graph_batch.ndata['node_tok_lens'].cpu().tolist()
        h = self.node_encoder(h, node_len)

        h = self.graph_encoder(graph_batch, h)  # [node_size, nhead, dims]
        h = torch.index_select(h, 0, var_node_index_in_node_batch)  # [var_node_size, nhead, dims]

        return h

    def get_func_emb(self, func_token_ids, attention_masks):
        # TODO https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        output = self.codebert(input_ids=func_token_ids, attention_mask=attention_masks,
                               output_hidden_states=True)  # 12层，每层是[batch, max_token_len, 768]
        func_emb = output.hidden_states[-1][:, 0]
        return func_emb

    def concat_wm_func_g_emb(self, batch, func_embs, wm_embs, g_embs, only_func_g_emb=False):
        concat_embs = []
        for batch_index, (g_emb_begin, g_emb_end) in enumerate(batch['func_map_var_position']):
            func_emb = func_embs[batch_index]

            if only_func_g_emb:
                wm_emb = None
            else:
                wm_emb = wm_embs[batch_index]

            for g_emb in g_embs[g_emb_begin:g_emb_end, :]:
                if only_func_g_emb:
                    concat_emb = torch.cat((func_emb, g_emb), dim=0)
                else:
                    concat_emb = torch.cat((func_emb, wm_emb, g_emb), dim=0)
                concat_embs.append(concat_emb)

        concat_embs = torch.stack(concat_embs, dim=0)
        return concat_embs

    def get_pre_var_logits(self, var_num, max_subtoken_len, emb):
        emb = self.catemb_to_var_decoder(emb)
        var_logits = torch.zeros(max_subtoken_len, var_num, self.config.vocab_size).to(self.config.device)
        hidden, cell = None, None
        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.var_decoder(emb, hidden, cell)
            var_logits[i] = var_logit

        # (subtoken_len, var_num, vocab) --> (var_num, subtoken_len, vocab)
        var_logits = torch.permute(var_logits, (1, 0, 2))
        return var_logits

    def get_rename_var_tok_embs(self, batch, var_selctions, pre_var_logits):
        var_selctions = F.gumbel_softmax(F.log_softmax(var_selctions, dim=-1), tau=0.1, hard=True)

        pre_var_tok_embs = self.node_encoder.get_var_emb(pre_var_logits)  # var_len, max_var_subtoken_len, tok_emb_size
        ori_var_ids = batch['var_tok_ids'][:, :pre_var_logits.shape[1]]
        ori_var_tok_embs = self.node_encoder.embed_node(ori_var_ids)

        rename_var_tok_embs = []
        for var_index in range(len(batch['vars'])):
            pre_var_tok_emb = pre_var_tok_embs[var_index]  # max_var_subtoken_len, tok_emb_size
            ori_var_tok_emb = ori_var_tok_embs[var_index]
            pre_ori_emb = torch.stack([pre_var_tok_emb, ori_var_tok_emb])  # 2, max_var_subtoken_len, tok_emb_size
            var_selection = torch.unsqueeze(var_selctions[var_index], dim=0)  # 1, 2
            rename_var_tok_emb = torch.matmul(var_selection,
                                              pre_ori_emb.reshape(2, -1))  # 1, max_var_subtoken_len, tok_emb_size
            rename_var_tok_emb = rename_var_tok_emb.reshape(pre_var_tok_emb.shape[0],
                                                            -1)  # max_var_subtoken_len, tok_emb_size
            rename_var_tok_embs.append(rename_var_tok_emb)
        rename_var_tok_embs = torch.stack(rename_var_tok_embs)  # var_len, max_var_subtoken_len, tok_emb_size

        return rename_var_tok_embs

    def get_rename_pos_and_var(self, batch):
        watermarks_class = []
        for batch_index, (g_begin, g_end) in enumerate(batch['func_map_var_position']):
            for _ in range(g_begin, g_end):
                watermarks_class.append(batch['watermarks_class'][batch_index])

        watermarks_class = torch.stack(watermarks_class, dim=0)

        ####
        func_emb = self.node_encoder(batch['func_token_ids'], batch['function_token_lens'])
        ####

        g_embs = self.encode_graph(batch['graph_batch'], batch['var_node_index_in_node_batch'],
                                   watermarks_class)
        if self.config.VarDecoder['cat_func_g_emb']:
            context = self.concat_wm_func_g_emb(batch=batch, func_embs=func_emb, wm_embs=None,
                                                g_embs=g_embs, only_func_g_emb=True)
        else:
            context = g_embs
        var_logits = self.var_decoder(batch=batch, context=context)


        wm_embs = self.wm_encoder(batch['watermarks'])
        h = self.get_g_embs(batch['graph_batch'], batch['var_node_index_in_node_batch'])  # [var_node_size, nhead, dims]
        # g_embs = h.reshape(h.shape[0], -1)  # [var_node_size, nhead x dims]
        g_embs = torch.mean(h, dim=1)

        wm_func_g_embs = self.concat_wm_func_g_emb(batch=batch,
                                                   func_embs=func_emb,
                                                   wm_embs=wm_embs,
                                                   g_embs=g_embs,
                                                   only_func_g_emb=False)
        var_selctions = self.var_selector(wm_func_g_embs)

        rename_var_tok_embs = self.get_rename_var_tok_embs(batch, var_selctions, var_logits)

        # func_gru_emb = self.func_gru(batch['func_token_ids'], batch['function_token_lens'])
        if self.config.use_distill_and_mse_loss:
            var_feats = g_embs
        else:
            var_feats = None
        return rename_var_tok_embs, var_logits, var_feats

    def inference(self, batch) -> Dict[str, str]:
        watermarks_class = []
        for batch_index, (g_begin, g_end) in enumerate(batch['func_map_var_position']):
            for _ in range(g_begin, g_end):
                watermarks_class.append(batch['watermarks_class'][batch_index])

        watermarks_class = torch.stack(watermarks_class, dim=0)

        ####
        func_emb = self.node_encoder(batch['func_token_ids'], batch['function_token_lens'])
        ####

        g_embs = self.encode_graph(batch['graph_batch'], batch['var_node_index_in_node_batch'],
                                   watermarks_class)
        if self.config.VarDecoder['cat_func_g_emb']:
            context = self.concat_wm_func_g_emb(batch=batch, func_embs=func_emb, wm_embs=None,
                                                g_embs=g_embs, only_func_g_emb=True)
        else:
            context = g_embs
        decoder_var_ids = self.var_decoder.inference(batch=batch, context=context)

        wm_embs = self.wm_encoder(batch['watermarks'])
        h = self.get_g_embs(graph_batch=batch['graph_batch'],
                            var_node_index_in_node_batch=batch['var_node_index_in_node_batch'])
        # g_embs = h.reshape(h.shape[0], -1)  # [var_node_size, nhead x dims]
        g_embs = torch.mean(h, dim=1)
        wm_func_g_embs = self.concat_wm_func_g_emb(batch=batch,
                                                   func_embs=func_emb,
                                                   wm_embs=wm_embs,
                                                   g_embs=g_embs,
                                                   only_func_g_emb=False)
        var_selctions = self.var_selector(wm_func_g_embs)

        ori_var_ids = batch['var_tok_ids']

        var_selctions = torch.argmax(var_selctions, dim=-1)

        pre_var_ids = []
        for var_index, select in enumerate(var_selctions):
            if select == 0:
                pre_var_ids.append(decoder_var_ids[var_index])
            else:
                pre_var_ids.append(ori_var_ids[var_index])

        sub_toks = get_var_from_var_id(self.config, batch['pre_var_tok_lens'], pre_var_ids)
        var_map = get_var_map(batch['vars'], sub_toks)
        return var_map


