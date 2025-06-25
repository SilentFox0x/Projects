from typing import Dict
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

from .var_decoder import VarDecoder, VarSelector, VarDecoderGru
from .gat import MyGatModel
from .wm_encoder import WMLinearEncoder
from utils import get_var_map, get_var_from_var_id


class Hiding(nn.Cell):
    def __init__(self, config, node_encoder, func_gru=None):
        super(Hiding, self).__init__()
        self.config = config
        self.wm_encoder = WMLinearEncoder(config.watermark_len, config.watermark_emb_dims)
        self.node_encoder = node_encoder
        self.func_gru = func_gru

        second_heads = 2 ** config.watermark_len
        self.graph_encoder = MyGatModel(
            layer_num=config.hiding_layers,
            first_in_feats=config.hiding['first_in_dim'],
            first_out_feats=config.hiding['first_out_dim'],
            first_heads=config.hiding['first_heads'],
            second_out_dim=config.hiding['second_out_dim'],
            second_heads=second_heads
        )

        if config.VarDecoder['decoder_type'] == 'lstm':
            self.var_decoder = VarDecoder(config)
        elif config.VarDecoder['decoder_type'] == 'gru':
            self.var_decoder = VarDecoderGru(config, node_encoder.embedding)
        else:
            raise TypeError('wrong decoder type')

        self.var_selector = VarSelector(config)

    def encode_graph(self, graphs, var_node_index_in_node_batch, watermarks_class):
        h = graphs.ndata['node_tok_ids']
        node_len = graphs.ndata['node_tok_lens'].asnumpy().tolist()
        h = self.node_encoder(h, node_len)
        h = self.graph_encoder(graphs, h)
        h = ops.Gather()(h, var_node_index_in_node_batch, 0)

        index_tensor = ops.Range()(0, len(watermarks_class), 1)
        h_selected = []
        for idx in index_tensor.asnumpy():
            h_selected.append(h[idx, watermarks_class[idx], :])
        h = ops.Stack(axis=0)(h_selected)
        return h

    def construct(self, watermarks_class, graph_batch, var_node_index_in_node_batch, pre_var_tok_lens,
                  var_tok_ids, var_tok_lens):
        g_emb = self.encode_graph(graph_batch, var_node_index_in_node_batch, watermarks_class)
        max_subtoken_len = max(pre_var_tok_lens)
        batch_size = watermarks_class.shape[0]
        var_logits = ops.Zeros()((max_subtoken_len, batch_size, self.config.vocab_size), mindspore.float32)
        hidden, cell = None, None

        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.var_decoder(g_emb, hidden, cell)
            var_logits[i] = var_logit

        var_logits = ops.Transpose()(var_logits, (1, 0, 2))
        return var_logits

    def get_g_embs(self, graph_batch, var_node_index_in_node_batch):
        h = graph_batch.ndata['node_tok_ids']
        node_len = graph_batch.ndata['node_tok_lens'].asnumpy().tolist()
        h = self.node_encoder(h, node_len)
        h = self.graph_encoder(graph_batch, h)
        h = ops.Gather()(h, var_node_index_in_node_batch, 0)
        return h

    def concat_wm_func_g_emb(self, batch, func_embs, wm_embs, g_embs, only_func_g_emb=False):
        concat_embs = []
        for i, (start, end) in enumerate(batch['func_map_var_position']):
            func_emb = func_embs[i]
            wm_emb = wm_embs[i] if not only_func_g_emb else None
            for g_emb in g_embs[start:end, :]:
                if only_func_g_emb:
                    concat = ops.Concat(0)((func_emb, g_emb))
                else:
                    concat = ops.Concat(0)((func_emb, wm_emb, g_emb))
                concat_embs.append(concat)
        return ops.Stack(0)(concat_embs)

    def get_pre_var_logits(self, var_num, max_subtoken_len, emb):
        emb = self.catemb_to_var_decoder(emb)
        var_logits = ops.Zeros()((max_subtoken_len, var_num, self.config.vocab_size), mindspore.float32)
        hidden, cell = None, None
        for i in range(max_subtoken_len):
            var_logit, hidden, cell = self.var_decoder(emb, hidden, cell)
            var_logits[i] = var_logit
        return ops.Transpose()(var_logits, (1, 0, 2))

    def get_rename_var_tok_embs(self, batch, var_selctions, pre_var_logits):
        var_selctions = ops.gumbel_softmax(ops.LogSoftmax(axis=-1)(var_selctions), tau=0.1, hard=True)
        pre_var_tok_embs = self.node_encoder.get_var_emb(pre_var_logits)
        ori_var_ids = batch['var_tok_ids'][:, :pre_var_logits.shape[1]]
        ori_var_tok_embs = self.node_encoder.embed_node(ori_var_ids)

        rename_var_tok_embs = []
        for i in range(len(batch['vars'])):
            pre = pre_var_tok_embs[i]
            ori = ori_var_tok_embs[i]
            combined = ops.Stack(0)((pre, ori))  # shape: (2, L, D)
            selector = ops.ExpandDims()(var_selctions[i], 0)  # shape: (1, 2)
            merged = ops.MatMul()(selector, combined.reshape(2, -1)).reshape(pre.shape)
            rename_var_tok_embs.append(merged)
        return ops.Stack(0)(rename_var_tok_embs)

    def get_rename_pos_and_var(self, batch):
        watermarks_class = ops.Stack(0)([
            batch['watermarks_class'][i] for i, (s, e) in enumerate(batch['func_map_var_position'])
            for _ in range(s, e)
        ])

        func_emb = self.node_encoder(batch['func_token_ids'], batch['function_token_lens'])
        g_embs = self.encode_graph(batch['graph_batch'], batch['var_node_index_in_node_batch'], watermarks_class)

        if self.config.VarDecoder['cat_func_g_emb']:
            context = self.concat_wm_func_g_emb(batch, func_emb, wm_embs=None, g_embs=g_embs, only_func_g_emb=True)
        else:
            context = g_embs

        var_logits = self.var_decoder(batch, context)

        wm_embs = self.wm_encoder(batch['watermarks'])
        g_embs_all = self.get_g_embs(batch['graph_batch'], batch['var_node_index_in_node_batch'])
        g_embs = ops.ReduceMean(keep_dims=False)(g_embs_all, 1)

        wm_func_g_embs = self.concat_wm_func_g_emb(batch, func_emb, wm_embs, g_embs, only_func_g_emb=False)
        var_selctions = self.var_selector(wm_func_g_embs)

        rename_var_tok_embs = self.get_rename_var_tok_embs(batch, var_selctions, var_logits)

        var_feats = g_embs if self.config.use_distill_and_mse_loss else None
        return rename_var_tok_embs, var_logits, var_feats

    def inference(self, batch) -> Dict[str, str]:
        watermarks_class = ops.Stack(0)([
            batch['watermarks_class'][i] for i, (s, e) in enumerate(batch['func_map_var_position'])
            for _ in range(s, e)
        ])
        func_emb = self.node_encoder(batch['func_token_ids'], batch['function_token_lens'])
        g_embs = self.encode_graph(batch['graph_batch'], batch['var_node_index_in_node_batch'], watermarks_class)

        if self.config.VarDecoder['cat_func_g_emb']:
            context = self.concat_wm_func_g_emb(batch, func_emb, wm_embs=None, g_embs=g_embs, only_func_g_emb=True)
        else:
            context = g_embs

        decoder_var_ids = self.var_decoder.inference(batch, context)

        wm_embs = self.wm_encoder(batch['watermarks'])
        g_embs_all = self.get_g_embs(batch['graph_batch'], batch['var_node_index_in_node_batch'])
        g_embs = ops.ReduceMean(keep_dims=False)(g_embs_all, 1)

        wm_func_g_embs = self.concat_wm_func_g_emb(batch, func_emb, wm_embs, g_embs, only_func_g_emb=False)
        var_selctions = self.var_selector(wm_func_g_embs)

        ori_var_ids = batch['var_tok_ids']
        var_selctions = ops.Argmax(axis=-1)(var_selctions)

        pre_var_ids = []
        for i in range(var_selctions.shape[0]):
            if var_selctions[i] == 0:
                pre_var_ids.append(decoder_var_ids[i])
            else:
                pre_var_ids.append(ori_var_ids[i])

        sub_toks = get_var_from_var_id(self.config, batch['pre_var_tok_lens'], pre_var_ids)
        var_map = get_var_map(batch['vars'], sub_toks)
        return var
