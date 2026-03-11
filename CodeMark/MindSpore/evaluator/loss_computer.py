import torch.nn as nn
import torch
import torch.nn.functional as F
from .nce_softmax_loss import NCESoftmaxLoss
from .recorder import LossRecorder
from typing import List


class LossComputer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.device

        self.watermark_loss_fn = nn.CrossEntropyLoss()
        self.feat_mse_loss_fn = nn.MSELoss()

        if self.config.use_infoNEC_loss:
            self.infoNCE_loss_fn = NCESoftmaxLoss(nce_t=self.config.nce_t)
        else:
            self.infoNCE_loss_fn = None

        # self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.tokenizer.pad_token_id)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)

        if self.config.use_cos_loss or self.config.use_var_ce_and_cos_loss:
            self.cos_loss_fn = nn.CosineEmbeddingLoss()
        else:
            self.cos_loss_fn = None

        self.recorder = LossRecorder(config)

        self.EPS = 1e-7

    def _get_w_loss(self, pre_watermark_class, watermarks_class):
        return self.watermark_loss_fn(pre_watermark_class, watermarks_class)

    def _get_w_loss_with_negative(self, pre_watermark_class, watermarks_class, negative_pre_watermark_class):
        temp = F.log_softmax(pre_watermark_class, dim=-1)
        # positive = self.watermark_loss_fn(pre_watermark_class, watermarks_class)
        positive = F.nll_loss(temp, watermarks_class)

        ones = torch.ones((negative_pre_watermark_class.shape[0], negative_pre_watermark_class.shape[1])).to(self.config.device)
        ones += self.EPS
        score = F.softmax(negative_pre_watermark_class, dim=-1)
        log_score = torch.log(ones - score)
        negative = F.nll_loss(log_score, watermarks_class)
        return positive + negative

    # 注意, 这里得到的pre_var_rep是使用Gumbel-Softmax得到的onehot计算的,
    # 好处是可以求导, 坏处是不稳定, 即pre_var_tok_id 可能!= torch.argmax(pre_var_onehot, dim=-1)
    # 所以如果计算sim(var, pre_var), 请使用 _computer_varclr_sim
    def _get_varclr_representation(self, var_tok_ids, var_tok_lens, pre_var_onehots, pre_var_tok_lens):
        batch_size = var_tok_ids.shape[0]
        # var_tok_ids [batch_size, max_node_token_len] -> var_ids_with_bos_eos [batch_size, max_node_token_len + 2]
        # 前加 bos, 后加 eos, 例如 [1234] -> [0,1234,2]
        var_ids_with_bos_eos = torch.ones((batch_size, var_tok_ids.shape[1] + 2),
                                          dtype=torch.long,
                                          device=self.config.device) * self.config.tokenizer.pad_token_id
        var_ids_with_bos_eos[:, 0] = self.config.tokenizer.bos_token_id
        for index, var_tok_len in enumerate(var_tok_lens):
            var_ids_with_bos_eos[index, 1:var_tok_len + 1] = var_tok_ids[index, :var_tok_len]
            var_ids_with_bos_eos[index, var_tok_len + 1] = self.config.tokenizer.eos_token_id

        var_attention_mask = (var_ids_with_bos_eos != self.config.tokenizer.get_pad_id()).float().to(self.config.device)
        var_rep = self.config.clr(var_ids_with_bos_eos, var_attention_mask)

        varclr_emb_weight = self.config.clr.transformer.base_model.embeddings.word_embeddings.weight
        emb2 = torch.matmul(pre_var_onehots, varclr_emb_weight)
        # +2是为了 bos, eos
        emb2_with_bos_eos = torch.ones((batch_size, emb2.shape[1] + 2, emb2.shape[2]), device=self.config.device) * \
                            self.config.tokenizer.pad_token_id
        emb2_with_bos_eos[:, 0, :] = varclr_emb_weight[self.config.tokenizer.bos_token_id, :]

        for index, pre_var_tok_len in enumerate(pre_var_tok_lens):
            emb2_with_bos_eos[index, 1:pre_var_tok_len + 1, :] = emb2[index, :pre_var_tok_len, :]
            emb2_with_bos_eos[index, pre_var_tok_len + 1, :] = varclr_emb_weight[self.config.tokenizer.eos_token_id, :]

        # 在var_attention_mask基础上为 prefix | suffix 增加1列
        diff_len = max(pre_var_tok_lens) - max(var_tok_lens)
        prefix_suffix_mask = torch.ones([batch_size, diff_len], dtype=torch.float, device=self.config.device)
        pre_var_attention_mask = torch.column_stack((prefix_suffix_mask, var_attention_mask))
        pre_var_rep = self.config.clr.compute_by_emb(emb2_with_bos_eos, pre_var_attention_mask)

        return var_rep, pre_var_rep

    def _compute_infoNCE_loss(self, varclr_rep1, varclr_rep2):
        return self.infoNCE_loss_fn(varclr_rep1, varclr_rep2)

    def _compute_infoNCE_loss_by_watermark_class(self, var_rep, pre_var_rep, watermarks_class):
        infoNCE_loss = []
        for class_type in range(4):
            selected_index = []
            for index, w_type in enumerate(watermarks_class):
                if w_type == class_type:
                    selected_index.append(index)
            if len(selected_index) > 0:
                selected_index = torch.tensor(selected_index, dtype=torch.long, device=self.device)

                select_emb1 = var_rep.index_select(0, selected_index).to(self.device)
                select_emb2 = pre_var_rep.index_select(0, selected_index).to(self.device)
                infoNCE_loss.append(self.infoNCE_loss_fn(select_emb1, select_emb2))
        return torch.stack(infoNCE_loss).mean()

    def _get_varclr_rep_by_tok_id(self, tok_ids, tok_lens):
        batch_size = tok_ids.shape[0]
        # var_tok_ids [batch_size, max_node_token_len] -> var_ids_with_bos_eos [batch_size, max_node_token_len + 2]
        # 前加 bos, 后加 eos, 例如 [1234] -> [0,1234,2]
        var_ids_with_bos_eos = torch.ones((batch_size, tok_ids.shape[1] + 2),
                                          dtype=torch.long,
                                          device=self.config.device) * self.config.tokenizer.pad_token_id
        var_ids_with_bos_eos[:, 0] = self.config.tokenizer.bos_token_id
        for index, var_tok_len in enumerate(tok_lens):
            var_ids_with_bos_eos[index, 1:var_tok_len + 1] = tok_ids[index, :var_tok_len]
            var_ids_with_bos_eos[index, var_tok_len + 1] = self.config.tokenizer.eos_token_id

        var_attention_mask = (var_ids_with_bos_eos != self.config.tokenizer.get_pad_id()).float().to(self.config.device)
        var_rep = self.config.clr(var_ids_with_bos_eos, var_attention_mask)
        return var_rep

    def _computer_varclr_sim(self, var_tok_ids, var_tok_lens, pre_var_tok_ids, pre_var_tok_lens):
        varclr_rep1 = self._get_varclr_rep_by_tok_id(var_tok_ids, var_tok_lens)
        varclr_rep2 = self._get_varclr_rep_by_tok_id(pre_var_tok_ids, pre_var_tok_lens)
        return F.cosine_similarity(varclr_rep1, varclr_rep2).mean()

    def _compute_var_ce_loss(self, pre_var_logits, var_tok_ids, var_tok_lens, ignore_n_subtoken):
        # vocab_dim = pre_var_logits.shape[-1]
        # pre_var_logits = pre_var_logits[:, 1:-1, :].reshape(-1, vocab_dim)
        # var_tok_ids = var_tok_ids.reshape(-1)
        # loss = self.ce_loss_fn(pre_var_logits, var_tok_ids)
        # return loss
        new_var_tok_ids, new_pre_var_logits = [], []
        for i, var_tok_len in enumerate(var_tok_lens):
            for j in range(var_tok_len):
                new_var_tok_ids.append(var_tok_ids[i][j])
                new_pre_var_logits.append(pre_var_logits[i][j+ignore_n_subtoken[i], :])
        new_var_tok_ids = torch.stack(new_var_tok_ids)
        new_pre_var_logits = torch.stack(new_pre_var_logits)
        return self.ce_loss_fn(new_pre_var_logits, new_var_tok_ids)

    def _computer_distill_loss(self, pre_var_logits, var_tok_lens, topk_idxs, topk_probs):
        new_pre_var_logits, new_topk_idxs, new_topk_probs = [], [], []
        for i, var_tok_len in enumerate(var_tok_lens):
            for j in range(var_tok_len):
                new_pre_var_logits.append(pre_var_logits[i][j, :])
                new_topk_idxs.append(topk_idxs[i][j, :])
                new_topk_probs.append(topk_probs[i][j, :])
        new_pre_var_logits = torch.stack(new_pre_var_logits)
        pre_var_probs = F.log_softmax(new_pre_var_logits, dim=-1)

        new_topk_idxs = torch.stack(new_topk_idxs)
        new_topk_probs = torch.stack(new_topk_probs)
        distill_loss = - (pre_var_probs.gather(dim=-1, index=new_topk_idxs) * new_topk_probs)
        # distill_loss = - (pre_var_probs * new_teacher_outputs).sum(dim=-1, keepdim=True)
        distill_loss = torch.mean(distill_loss)
        return distill_loss

    def _computer_triplet_loss(self, var_rep, pre_var_rep, watermarks_class=None, use_w_class=False):
        if use_w_class:
            loss = []
            for class_type in range(4):
                selected_index = []
                for index, w_type in enumerate(watermarks_class):
                    if w_type == class_type:
                        selected_index.append(index)
                if len(selected_index) > 0:
                    selected_index = torch.tensor(selected_index, dtype=torch.long, device=self.device)

                    select_var_rep = var_rep.index_select(0, selected_index).to(self.device)
                    select_pre_var_rep = pre_var_rep.index_select(0, selected_index).to(self.device)
                    negative_pointer = select_pre_var_rep
                    indices = torch.randperm(negative_pointer.size()[0])
                    negative = negative_pointer[indices]
                    loss.append(self.triplet_loss_fn(select_var_rep, select_pre_var_rep, negative))
            return torch.stack(loss).mean()
        else:
            negative_pointer = var_rep
            indices = torch.randperm(negative_pointer.size()[0])
            negative = negative_pointer[indices]
            # anchor, positive, negative
            return self.triplet_loss_fn(var_rep, pre_var_rep, negative)

    def _computer_cos_loss(self, var_rep, pre_var_rep):
        batch_size = var_rep.shape[0]
        target = [1] * batch_size
        target = torch.tensor(target, dtype=torch.long).to(var_rep.device)
        return self.cos_loss_fn(var_rep, pre_var_rep, target)

    def _computer_var_ce_and_cos_loss(
            self, var_tok_ids, pre_var_logits, pre_var_rep, watermarks_class, var_tok_lens,
            ignore_n_subtoken):
        ce_loss = self._compute_var_ce_loss(pre_var_logits=pre_var_logits, var_tok_ids=var_tok_ids,
                                            var_tok_lens=var_tok_lens, ignore_n_subtoken=ignore_n_subtoken)

        negative_var_cos_loss = []
        for class_type in range(4):
            selected_index = []
            for index, w_type in enumerate(watermarks_class):
                if w_type == class_type:
                    selected_index.append(index)
            if len(selected_index) > 0:
                selected_index = torch.tensor(selected_index, dtype=torch.long, device=self.device)

                select_pre_var_rep = pre_var_rep.index_select(0, selected_index).to(self.device)
                negative_pointer = select_pre_var_rep
                indices = torch.randperm(negative_pointer.size()[0])
                negative = negative_pointer[indices]
                sample_size = select_pre_var_rep.shape[0]
                target = torch.tensor([-1] * sample_size, dtype=torch.long).to(select_pre_var_rep.device)
                negative_var_cos_loss.append(self.cos_loss_fn(select_pre_var_rep, negative, target))
        negative_var_cos_loss = torch.stack(negative_var_cos_loss).mean()
        return ce_loss + negative_var_cos_loss

    def _compute_var_ce_and_triplet_loss(
            self, var_tok_ids, var_rep, pre_var_rep, pre_var_logits, watermarks_class, var_tok_lens,
            ignore_n_subtoken):
        ce_loss = self._compute_var_ce_loss(pre_var_logits=pre_var_logits, var_tok_ids=var_tok_ids,
                                            var_tok_lens=var_tok_lens, ignore_n_subtoken=ignore_n_subtoken)
        triplet_loss = self._computer_triplet_loss(var_rep=var_rep, pre_var_rep=pre_var_rep,
                                                   watermarks_class=watermarks_class,
                                                   use_w_class=True)
        return self.config.var_ce_loss_weight * ce_loss + self.config.var_triplet_loss_weight * triplet_loss

    def _computer_g_feat_mse_loss(self, ori_g_feats, pre_g_feats, func_map_var_position):
        tot_mse_losses = 0
        if self.config.granularity == 'var':
            for batch_index, (g_emb_begin, g_emb_end) in enumerate(func_map_var_position):
                var_nums = g_emb_end - g_emb_begin
                for i in range(g_emb_begin, g_emb_end):
                    tot_mse_losses += 1/var_nums * self.feat_mse_loss_fn(ori_g_feats[i], pre_g_feats[i])
        elif self.config.granularity == 'func':
            for batch_index, (g_emb_begin, g_emb_end) in enumerate(func_map_var_position):
                ori_fun_feat = torch.mean(ori_g_feats[g_emb_begin:g_emb_end], dim=0)
                pre_fun_feat = torch.mean(pre_g_feats[g_emb_begin:g_emb_end], dim=0)
                tot_mse_losses += self.feat_mse_loss_fn(ori_fun_feat, pre_fun_feat)
        else:
            raise Exception('error feat mse loss')
        return tot_mse_losses

    def get_loss(self, pre_watermark_class, watermarks_class, watermarks,
                 var_tok_ids, var_tok_lens, pre_var_tok_lens,
                 pre_var_onehots, pre_var_logits, pre_var_tok_ids,
                 epoch_index: int, ignore_n_subtoken,
                 topk_idxs, topk_probs, pre_vars: List[str],
                 func_emb=None, pre_func_emb=None, func_map_var_position=None):
        w_loss, var_sim, infoNCE_loss, var_ce_loss, triplet_loss, perplexity, var_cos_loss = \
            None, None, None, None, None, None, None
        var_ce_and_cos_loss, var_ce_and_triplet_loss, distill_loss, mse_loss = None, None, None, None

        w_loss = self._get_w_loss(pre_watermark_class, watermarks_class)
        # w_loss = self._get_w_loss_with_negative(pre_watermark_class, watermarks_class, negative_pre_watermark_class)

        # var_rep, pre_var_rep = self._get_varclr_representation(
        #     var_tok_ids=var_tok_ids, var_tok_lens=var_tok_lens,
        #     pre_var_onehots=pre_var_onehots, pre_var_tok_lens=pre_var_tok_lens)
        # var_sim = self._computer_varclr_sim(
        #     var_tok_ids=var_tok_ids, var_tok_lens=var_tok_lens,
        #     pre_var_tok_ids=pre_var_tok_ids, pre_var_tok_lens=pre_var_tok_lens)

        # infoNCE_loss = \
        #     self._compute_infoNCE_loss_by_watermark_class(var_rep, pre_var_rep, watermarks_class)

        if self.config.use_var_ce_and_cos_loss:
            var_ce_and_cos_loss = \
                self._computer_var_ce_and_cos_loss(
                    var_tok_ids=var_tok_ids, pre_var_logits=pre_var_logits,
                    pre_var_rep=pre_var_rep, watermarks_class=watermarks_class,
                    var_tok_lens=var_tok_lens, ignore_n_subtoken=ignore_n_subtoken)
            loss = self.config.w_loss_weight * w_loss + var_ce_and_cos_loss
        elif self.config.use_var_ce_and_triplet_loss:
            var_ce_and_triplet_loss = self._compute_var_ce_and_triplet_loss(
                var_tok_ids=var_tok_ids, var_rep=var_rep, pre_var_rep=pre_var_rep,
                pre_var_logits=pre_var_logits, watermarks_class=watermarks_class,
                var_tok_lens=var_tok_lens, ignore_n_subtoken=ignore_n_subtoken)
            loss = self.config.w_loss_weight * w_loss + var_ce_and_triplet_loss
        elif self.config.use_var_ce_loss and epoch_index >= self.config.begin_use_var_ce_loss_epoch:
            var_ce_loss = self._compute_var_ce_loss(
                pre_var_logits=pre_var_logits, var_tok_ids=var_tok_ids,
                var_tok_lens=var_tok_lens, ignore_n_subtoken=ignore_n_subtoken)
            loss = self.config.w_loss_weight * w_loss + self.config.var_ce_loss_weight * var_ce_loss
        elif self.config.use_triplet_loss and epoch_index >= self.config.begin_use_triplet_loss_epoch:
            triplet_loss = self._computer_triplet_loss(var_rep, pre_var_rep, watermarks_class, True)
            loss = self.config.w_loss_weight * w_loss + self.config.triplet_loss_weight * triplet_loss
        elif self.config.use_cos_loss and epoch_index >= self.config.begin_use_var_cos_loss_epoch:
            var_cos_loss = self._computer_cos_loss(var_rep=var_rep, pre_var_rep=pre_var_rep)
            loss = self.config.w_loss_weight * w_loss + self.config.var_cos_weight * var_cos_loss
        elif self.config.use_distill_loss and epoch_index >= self.config.begin_use_distill_loss_epoch:
            distill_loss = self._computer_distill_loss(pre_var_logits, var_tok_lens, topk_idxs, topk_probs)
            loss = self.config.w_loss_weight * w_loss + self.config.distill_loss_weight * distill_loss
        elif self.config.use_distill_and_mse_loss and epoch_index >= self.config.begin_epoch:
            distill_loss = self._computer_distill_loss(pre_var_logits, var_tok_lens, topk_idxs, topk_probs)
            mse_loss = self._computer_g_feat_mse_loss(func_emb, pre_func_emb, func_map_var_position)
            # if mse_loss < 0.005:
            # if mse_loss < 1:
            #     self.config.cur_feat_mse_loss_weight = 0.0
            # else:
            #     self.config.cur_feat_mse_loss_weight = self.config.feat_mse_loss_weight
            self.config.cur_feat_mse_loss_weight = self.config.feat_mse_loss_weight
            loss = self.config.w_loss_weight * w_loss + self.config.distill_loss_weight * distill_loss \
                   + self.config.cur_feat_mse_loss_weight * mse_loss
        else:
            loss = w_loss

        self.recorder.record_by_multi_class(pre_watermark_class=pre_watermark_class,
                                            watermarks=watermarks,
                                            loss=loss, w_loss=w_loss,
                                            infoNCE_loss=infoNCE_loss, var_ce_loss=var_ce_loss,
                                            triplet_loss=triplet_loss,
                                            var_sim=var_sim, perplexity=perplexity,
                                            var_cos_loss=var_cos_loss, var_ce_and_cos_loss=var_ce_and_cos_loss,
                                            var_ce_and_triplet_loss=var_ce_and_triplet_loss,
                                            distill_loss=distill_loss,
                                            pre_vars=pre_vars,
                                            mse_loss=mse_loss)

        return loss
