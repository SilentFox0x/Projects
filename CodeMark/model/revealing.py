import torch.nn as nn
from dataset import VarReplaceHelper
from typing import List
import torch
from .gat import MyGatModel
from evaluator import get_watermarks_from_w_class
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
class Revealing(nn.Module):
    def __init__(self, config, node_encoder, func_gru=None):
        super(Revealing, self).__init__()
        self.config = config

        self.node_encoder = node_encoder
        self.func_gru = func_gru
        # self.codebert = codebert

        self.graph_encoder = MyGatModel(layer_num=self.config.hiding_layers,
                                        first_in_feats=self.config.revealing['first_in_dim'],
                                        first_out_feats=self.config.revealing['first_out_dim'],
                                        first_heads=self.config.revealing['first_heads'],
                                        second_out_dim=self.config.revealing['second_out_dim'],
                                        second_heads=self.config.revealing['second_heads'])

        watermark_classes = 1 << self.config.watermark_len
        # input_dims = self.config.node_encoder_out_dims * watermark_classes
        #
        # self.watermark_decoder = nn.Sequential(
        #     nn.Linear(input_dims,
        #               self.config.watermark_decoder_hidden_dims),
        #     nn.LeakyReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(self.config.watermark_decoder_hidden_dims, watermark_classes),
        # )
        self.watermark_decoder = nn.Sequential(
            nn.Linear(self.config.watermark_decoder_in_dims, self.config.watermark_decoder_hidden_dims),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.config.watermark_decoder_hidden_dims, watermark_classes),
        )

    def encode_graph(self, var_position_in_node_batch: List[List[VarReplaceHelper]], graphs, pre_var_emb,
                     var_node_index_in_node_batch, pre_var_tok_lens):
        h = graphs.ndata['node_tok_ids']
        h = self.node_encoder.embed_node(h)
        for node_index, var_rep_hels in enumerate(var_position_in_node_batch):
            for vrh in var_rep_hels:
                pre_var_tok_len = pre_var_tok_lens[vrh.var_index_in_vars]
                # TODO 这个判断应该可以删了？
                if vrh.begin + pre_var_tok_len > h.shape[1]:
                    continue
                h[node_index, vrh.begin:vrh.begin + pre_var_tok_len, :] = \
                    pre_var_emb[vrh.var_index_in_vars, :pre_var_tok_len, :]

        node_len = graphs.ndata['node_tok_lens'].cpu().tolist()
        h = self.node_encoder.get_lstm_output(h, node_len)

        h = self.graph_encoder(graphs, h)  # [node_size, nhead, dims]
        h = torch.index_select(h, dim=0, index=var_node_index_in_node_batch)  # [var_node_size, nhead, dims]

        return h

    def forward(self, var_position_in_node_batch, graph_batch, pre_var_logits,
                var_node_index_in_node_batch, pre_var_tok_lens, watermarks_class):
        pre_var_tok_emb = self.node_encoder.get_var_emb(pre_var_logits)
        g_emb = self.encode_graph(var_position_in_node_batch, graph_batch, pre_var_tok_emb,
                                   var_node_index_in_node_batch, pre_var_tok_lens)

        pre_watermark_class = self.watermark_decoder(g_emb)
        return pre_watermark_class

    # def get_func_embs_after_rename(self, batch, rename_var_tok_embs):
    #     func_token_embs = self.codebert.roberta.embeddings.word_embeddings(batch['func_token_ids'])
    #
    #     for func_index, var_rep_hels in enumerate(batch['var_position_in_func_batch']):
    #         for vrh in var_rep_hels:
    #             pre_var_tok_len = batch['pre_var_tok_lens'][vrh.var_index_in_vars]
    #             func_token_embs[func_index, vrh.begin:vrh.begin + pre_var_tok_len, :] = \
    #                 rename_var_tok_embs[vrh.var_index_in_vars, :pre_var_tok_len, :]
    #
    #     output = self.codebert(inputs_embeds=func_token_embs, attention_mask=batch['attention_masks'],
    #                            output_hidden_states=True)  # 12层，每层是[batch, max_token_len, 768]
    #     func_embs = output.hidden_states[-1][:, 0]
    #     return func_embs

    def get_func_embs_after_rename(self, func_token_ids, function_token_lens, var_position_in_func_batch,
                                   pre_var_emb, pre_var_tok_lens):
        h = self.node_encoder.embed_node(func_token_ids)
        for func_index, var_rep_hels in enumerate(var_position_in_func_batch):
            for vrh in var_rep_hels:
                pre_var_tok_len = pre_var_tok_lens[vrh.var_index_in_vars]
                h[func_index, vrh.begin:vrh.begin + pre_var_tok_len, :] = \
                    pre_var_emb[vrh.var_index_in_vars, :pre_var_tok_len, :]

        h = self.func_gru.get_gru_output(h, function_token_lens)
        return h

    def get_(self, batch, rename_var_tok_embs):
        h = self.encode_graph(var_position_in_node_batch=batch['var_position_in_node_batch'],
                                  graphs=batch['graph_batch'],
                                  pre_var_emb=rename_var_tok_embs,
                                  var_node_index_in_node_batch=batch['var_node_index_in_node_batch'],
                                  pre_var_tok_lens=batch['pre_var_tok_lens'])

        # g_embs = h.reshape(h.shape[0], -1)
        g_embs = torch.mean(h, dim=1)


        # pre_wm_emb = self.merge_func_graph_emb(batch, func_embs, g_embs)

        pre_wm_embs = []
        for batch_index, (g_emb_begin, g_emb_end) in enumerate(batch['func_map_var_position']):
            pre_wm_emb = torch.max(g_embs[g_emb_begin:g_emb_end], dim=0)
            pre_wm_embs.append(pre_wm_emb.values)
        pre_wm_embs = torch.stack(pre_wm_embs, dim=0)

        pre_watermark_class = self.watermark_decoder(pre_wm_embs)

        # pre_func_gru_emb = self.get_func_embs_after_rename(func_token_ids=batch['func_token_ids'],
        #                                             function_token_lens=batch['function_token_lens'],
        #                                             var_position_in_func_batch=batch['var_position_in_func_batch'],
        #                                             pre_var_emb=rename_var_tok_embs,
        #                                             pre_var_tok_lens=batch['pre_var_tok_lens'])
        if self.config.use_distill_and_mse_loss:
            pre_feats = g_embs
        else:
            pre_feats = None

        return pre_watermark_class, pre_feats

    def inference(self, batch):
        graphs, var_node_index_in_node_batch = batch['graph_batch'], batch['var_node_index_in_node_batch']
        h = graphs.ndata['node_tok_ids']
        node_len = graphs.ndata['node_tok_lens'].cpu().tolist()
        h = self.node_encoder(h, node_len)

        h = self.graph_encoder(graphs, h)  # [node_size, nhead, dims]
        h = torch.index_select(h, dim=0, index=var_node_index_in_node_batch)
        g_embs = torch.mean(h, dim=1)

        pre_wm_embs = []
        for batch_index, (g_emb_begin, g_emb_end) in enumerate(batch['func_map_var_position']):
            pre_wm_emb = torch.max(g_embs[g_emb_begin:g_emb_end], dim=0)
            pre_wm_embs.append(pre_wm_emb.values)
        pre_wm_embs = torch.stack(pre_wm_embs, dim=0)

        pre_watermark_class = self.watermark_decoder(pre_wm_embs)
        mutil_subgraph_watermark = get_watermarks_from_w_class(pre_watermark_class=pre_watermark_class)
        return mutil_subgraph_watermark
