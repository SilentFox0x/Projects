from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from .shannon_entropy import entropy
from dataset import watermark_class_to_watermark


class LossRecorder:
    def __init__(self, config):
        self.batch_num = 0
        self.sigmoid = nn.Sigmoid()
        self.config = config
        self.watermark_len = config.watermark_len

        # loss
        self.tot_loss = 0
        self.tot_w_loss = 0
        # self.tot_infoNCE_loss = 0
        self.tot_var_ce_loss = 0
        self.tot_triplet_loss = 0
        self.tot_var_cos_loss = 0
        self.tot_var_ce_and_cos_loss = 0
        self.tot_var_ce_and_triplet_loss = 0
        self.tot_distill_loss = 0
        self.tot_mse_loss = 0

        # metric
        self.tot_watermark_acc = 0
        # self.tot_perplexity = 0
        self.tot_var_sim = 0
        # self.tot_watermark_acc_by_bit = []
        # self.tot_watermark_TP_FP_by_bit = []
        # for i in range(self.watermark_len):
        #     self.tot_watermark_acc_by_bit.append(0)
        #     self.tot_watermark_TP_FP_by_bit.append({'0TP': 0, '0FP': 0, '1TP': 0, '1FP': 0})

        self.current_w_acc = 0
        self.current_var_sim = 0

        self.pre_vars = []

    def _compute_watermark(self, pre_watermark, watermark_labels):
        pre_watermark = torch.round(self.sigmoid(pre_watermark))
        self.tot_watermark_acc += (pre_watermark == watermark_labels).float().mean().item()
        self.current_w_acc = (pre_watermark == watermark_labels).float().mean().item()
        # for bit_index in range(self.watermark_len):
        #     self.tot_watermark_acc_by_bit[bit_index] += (
        #             pre_watermark[:, bit_index] == watermark_labels[:, bit_index]).float().mean().item()
        #
        #     bit_predict = pre_watermark[:, bit_index].long()
        #     bit_label = watermark_labels[:, bit_index].long()
        #     true_false = bit_predict == bit_label
        #     self.tot_watermark_TP_FP_by_bit[bit_index]['0TP'] += \
        #         len(['' for p, tf in zip(bit_predict, true_false) if p == 0 and tf == True])
        #     self.tot_watermark_TP_FP_by_bit[bit_index]['0FP'] += \
        #         len(['' for p, tf in zip(bit_predict, true_false) if p == 0 and tf == False])
        #     self.tot_watermark_TP_FP_by_bit[bit_index]['1TP'] += \
        #         len(['' for p, tf in zip(bit_predict, true_false) if p == 1 and tf == True])
        #     self.tot_watermark_TP_FP_by_bit[bit_index]['1FP'] += \
        #         len(['' for p, tf in zip(bit_predict, true_false) if p == 1 and tf == False])

    def record(self, loss=None, w_loss=None, infoNCE_loss=None, var_ce_loss=None,
               triplet_loss=None,
               pre_watermark=None, watermarks=None, perplexity=None, var_sim=None,
               var_cos_loss=None, var_ce_and_cos_loss=None, var_ce_and_triplet_loss=None,
               distill_loss=None, pre_vars=None, mse_loss=None):
        self.batch_num += 1

        if loss is not None:
            self.tot_loss += loss.item()
        if w_loss is not None:
            self.tot_w_loss += w_loss.item()
        # if infoNCE_loss is not None:
        #     self.tot_infoNCE_loss += infoNCE_loss.item()
        if var_ce_loss is not None:
            self.tot_var_ce_loss += var_ce_loss.item()
        if triplet_loss is not None:
            self.tot_triplet_loss += triplet_loss.item()

        if pre_watermark is not None:
            self._compute_watermark(pre_watermark=pre_watermark, watermark_labels=watermarks)
        # if perplexity is not None:
        #     self.tot_perplexity += perplexity.item()
        if var_sim is not None:
            self.tot_var_sim += var_sim.item()
            self.current_var_sim = var_sim.item()
        if var_cos_loss is not None:
            self.tot_var_cos_loss += var_cos_loss.item()
        if var_ce_and_cos_loss is not None:
            self.tot_var_ce_and_cos_loss += var_ce_and_cos_loss.item()
        if var_ce_and_triplet_loss is not None:
            self.tot_var_ce_and_triplet_loss += var_ce_and_triplet_loss.item()
        if distill_loss is not None:
            self.tot_distill_loss += distill_loss.item()
        if mse_loss is not None:
            self.tot_mse_loss += mse_loss.item()

        if pre_vars is not None:
            self.pre_vars += pre_vars

    def record_by_multi_class(self, pre_watermark_class, watermarks,
                              loss, w_loss, infoNCE_loss, var_ce_loss,
                              triplet_loss,
                              perplexity, var_sim, var_cos_loss, var_ce_and_cos_loss,
                              var_ce_and_triplet_loss, distill_loss, pre_vars, mse_loss):
        pre_watermark = get_watermarks_from_w_class(pre_watermark_class)

        pre_watermark = torch.tensor(pre_watermark, dtype=torch.float).to(pre_watermark_class.device)
        return self.record(loss=loss, w_loss=w_loss, infoNCE_loss=infoNCE_loss, var_ce_loss=var_ce_loss,
                           pre_watermark=pre_watermark, watermarks=watermarks, triplet_loss=triplet_loss,
                           perplexity=perplexity, var_sim=var_sim, var_cos_loss=var_cos_loss,
                           var_ce_and_cos_loss=var_ce_and_cos_loss, var_ce_and_triplet_loss=var_ce_and_triplet_loss,
                           distill_loss=distill_loss, pre_vars=pre_vars, mse_loss=mse_loss)

    def get_results(self) -> Dict:
        avg_loss = self.tot_loss / self.batch_num
        avg_w_loss = self.tot_w_loss / self.batch_num
        # avg_clr_loss = self.tot_clr_loss / self.batch_num
        avg_ce_loss = self.tot_var_ce_loss / self.batch_num
        avg_triplet_loss = self.tot_triplet_loss / self.batch_num
        avg_var_cos_loss = self.tot_var_cos_loss / self.batch_num
        avg_var_ce_and_cos_loss = self.tot_var_ce_and_cos_loss / self.batch_num
        avg_var_ce_and_triplet_loss = self.tot_var_ce_and_triplet_loss / self.batch_num
        avg_distill_loss = self.tot_distill_loss / self.batch_num
        avg_mse_loss = self.tot_mse_loss / self.batch_num

        avg_var_sim = self.tot_var_sim / self.batch_num
        avg_watermark_acc = self.tot_watermark_acc / self.batch_num
        # avg_perplexity = self.tot_perplexity / self.batch_num

        shannon_entropy = entropy(self.pre_vars)

        return {
            "loss": avg_loss,
            "w_loss": avg_w_loss,
            # "clr_loss": avg_clr_loss,
            "ce_loss": avg_ce_loss,
            "triplet_loss": avg_triplet_loss,
            "var_cos_loss": avg_var_cos_loss,
            "var_ce_and_cos_loss": avg_var_ce_and_cos_loss,
            "var_ce_and_triplet_loss": avg_var_ce_and_triplet_loss,
            "distill_loss": avg_distill_loss,
            "var_sim": avg_var_sim,
            "watermark_acc": avg_watermark_acc,
            # "perplexity": avg_perplexity,
            "shannon_entropy": shannon_entropy,
            "mse_loss": avg_mse_loss
        }

    def print_bit_acc(self) -> None:
        print('deprecated API')
        # for bit_index in range(self.watermark_len):
        #     self.config.logger.info(
        #         f'{bit_index} bit: {self.tot_watermark_acc_by_bit[bit_index] / self.batch_num:.4f}, '
        #         f'{self.tot_watermark_TP_FP_by_bit[bit_index]}')

    def print_metrics(self, prefix='') -> None:
        results = self.get_results()
        string_builder = prefix
        for key, value in results.items():
            if key.endswith('_loss') and value < 1e-7:
                continue
            string_builder += f'{key}: {value:.4f} | '
        self.config.logger.info(string_builder)

    def get_current_metrics(self):
        return self.current_w_acc, self.current_var_sim


def get_watermarks_from_w_class(pre_watermark_class):
    pre_watermark = []
    for item in torch.argmax(pre_watermark_class, dim=1):
        w_bits = watermark_class_to_watermark(item)
        pre_watermark.append(w_bits)
    return pre_watermark
