import argparse
from dataset import MyDataset, graph_collate, WatermarkFactory, CsnDataset, csn_collate
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CodeMark
import torch
from tqdm import tqdm
from configs import Config
import numpy as np
import random
import os
from utils import get_parameter_number
from evaluator import LossComputer
from typing import Tuple


def set_random_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True


def train(config, model: CodeMark, dataloader, optimizer, epoch_index: int):
    model.train()
    loss_computer = LossComputer(config)
    progress = tqdm(dataloader)
    for i, batch in enumerate(progress):
        optimizer.zero_grad()
        pre_watermark_class, pre_var_logits, feats, pre_feats = model(batch)
        pre_var_onehots = None
        pre_var_ids = None
        pre_vars = None
        # pre_var_logits=None

        # if i == 59:
        #     print(i)

        # pre_watermark_class, pre_var_logits, pre_var_ids, pre_var_onehots, pre_vars = \
        #     model(watermarks_class=batch['watermarks_class'],
        #           graph_batch=batch['graph_batch'],
        #           var_node_index_in_node_batch=batch['var_node_index_in_node_batch'],
        #           var_tok_lens=batch['var_tok_lens'],
        #           pre_var_tok_lens=batch['pre_var_tok_lens'],
        #           var_position_in_node_batch=batch['var_position_in_node_batch'],
        #           var_tok_ids=batch['var_tok_ids'])

        loss = loss_computer.get_loss(pre_watermark_class=pre_watermark_class,
                                      watermarks_class=batch['watermarks_class'],
                                      watermarks=batch['watermarks'],
                                      var_tok_ids=batch['var_tok_ids'],
                                      var_tok_lens=batch['var_tok_lens'],
                                      pre_var_tok_lens=batch['pre_var_tok_lens'],
                                      pre_var_onehots=pre_var_onehots,
                                      pre_var_logits=pre_var_logits,
                                      pre_var_tok_ids=pre_var_ids,
                                      epoch_index=epoch_index,
                                      ignore_n_subtoken=batch['ignore_n_subtoken'],
                                      topk_idxs=batch['topk_idxs'],
                                      topk_probs=batch['topk_probs'],
                                      pre_vars=pre_vars,
                                      func_emb=feats,
                                      pre_func_emb=pre_feats,
                                      func_map_var_position=batch['func_map_var_position'])

        loss.backward()
        optimizer.step()

        results = loss_computer.recorder.get_results()
        if config.use_var_ce_and_cos_loss:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{results['var_ce_and_cos_loss']:.2f}, "
        elif config.use_var_ce_and_triplet_loss:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{results['var_ce_and_triplet_loss']:.2f}, "
        elif config.use_var_ce_loss and epoch_index >= config.begin_use_var_ce_loss_epoch:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{config.var_ce_loss_weight}*{results['ce_loss']:.2f}, "
        elif config.use_triplet_loss and epoch_index >= config.begin_use_triplet_loss_epoch:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{config.triplet_loss_weight}*{results['triplet_loss']:.2f}, "
        elif config.use_cos_loss and epoch_index >= config.begin_use_var_cos_loss_epoch:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{config.var_cos_weight}*{results['var_cos_loss']:.2f}, "
        elif config.use_distill_loss and epoch_index >= config.begin_use_distill_loss_epoch:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{config.distill_loss_weight}*{results['distill_loss']:.2f}, "
        elif config.use_distill_and_mse_loss and epoch_index >= config.begin_epoch:
            des = f"loss: {results['loss']:.2f}={config.w_loss_weight}*{results['w_loss']:.2f}+" \
                  f"{config.distill_loss_weight:.2f}*{results['distill_loss']:.2f}+" \
                  f"{config.cur_feat_mse_loss_weight:.2f}*{results['mse_loss']:.6f}, "
        else:
            des = f"loss: {results['loss']:.2f}={results['w_loss']:.2f}+"
        des += f"watermark_acc: {results['watermark_acc']:.2f}, var_sim: {results['var_sim']:.2f}"
        progress.set_description(des)
    loss_computer.recorder.print_metrics()


def valid(config, model, dataloader, epoch_index: int) -> Tuple[float, float]:
    config.logger.info('testing...')
    model.eval()
    progress = tqdm(dataloader)
    loss_computer = LossComputer(config)
    with torch.no_grad():
        for i, batch in enumerate(progress):
            pre_var_onehots = None
            pre_var_ids = None
            pre_vars = None

            pre_watermark_class, pre_var_logits, func_gru_emb, pre_func_gru_emb = model(batch)
            loss_computer.get_loss(pre_watermark_class=pre_watermark_class,
                                   watermarks_class=batch['watermarks_class'],
                                   watermarks=batch['watermarks'],
                                   var_tok_ids=batch['var_tok_ids'],
                                   var_tok_lens=batch['var_tok_lens'],
                                   pre_var_tok_lens=batch['pre_var_tok_lens'],
                                   pre_var_onehots=pre_var_onehots,
                                   pre_var_logits=pre_var_logits,
                                   pre_var_tok_ids=pre_var_ids,
                                   epoch_index=epoch_index,
                                   ignore_n_subtoken=batch['ignore_n_subtoken'],
                                   topk_idxs=batch['topk_idxs'],
                                   topk_probs=batch['topk_probs'],
                                   pre_vars=pre_vars,
                                   func_emb=func_gru_emb,
                                   pre_func_emb=pre_func_gru_emb,
                                   func_map_var_position=batch['func_map_var_position'])

    loss_computer.recorder.print_metrics(prefix='\t\t')
    results = loss_computer.recorder.get_results()
    return results['watermark_acc'], results['var_sim']


def main():
    set_random_seed()
    config = Config(args.config_path)
    config.language = args.lang
    config.device = 'cuda:' + args.device
    config.watermark_len = args.watermark_len
    config.VarSelector['mask_probability'] = args.var_selector_mask_probability
    config.VarDecoder['substitute_mask_probability'] = args.substitute_mask_probability
    config.VarDecoder['topk'] = args.topk
    if config.mode != 'test':
        config.print_config()

    watermark_factory = WatermarkFactory(config.watermark_len)

    train_dataset = CsnDataset(filepath=config.dataset_path[config.language]['train_data_path'],
                               temperature=config.temperature,
                               logger=config.logger,
                               use_type_or_text=config.use_type_or_text,
                               max_samples=args.max_samples)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True,
                              num_workers=0,
                              collate_fn=lambda batch: csn_collate(batch, config, watermark_factory))
    valid_dataset = CsnDataset(filepath=config.dataset_path[config.language]['valid_data_path'],
                               temperature=config.temperature,
                               logger=config.logger,
                               use_type_or_text=config.use_type_or_text,
                               max_samples=args.max_samples)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.test_batch_size, shuffle=False,
                              num_workers=0,
                              collate_fn=lambda batch: csn_collate(batch, config, watermark_factory))

    config.logger.info('construct model...')
    # model = MyModel(config)
    model = CodeMark(config)
    model = model.to(config.device)
    config.logger.info(model)
    config.logger.info(get_parameter_number(model))

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    # embedding_params = list(map(id, model.node_encoder.embedding.parameters()))
    # other_params = filter(lambda p: id(p) not in embedding_params, model.parameters())
    # optimizer = Adam([
    #     {'params': other_params, 'lr': config.lr},
    #     {'params': model.node_encoder.embedding.parameters(), 'lr': 0.1 * config.lr}
    # ])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=5)

    best_w_acc, best_var_sim = 0.5, 0.38
    # best_all = best_w_acc + best_var_sim
    for epoch_index in range(1, config.epoch + 1):
        config.logger.info('epoch: ' + str(epoch_index))

        all_best_model_path = os.path.join(config.model_save_path,
                                           f'{config.timestamp}-E{str(epoch_index)}-all-best.pth')
        w_acc_best_model_path = os.path.join(config.model_save_path,
                                             f'{config.timestamp}-E{str(epoch_index)}-w-acc-best.pth')
        var_sim_best_model_path = os.path.join(config.model_save_path,
                                               f'{config.timestamp}-E{str(epoch_index)}-varsim-best.pth')

        train(config=config, model=model, dataloader=train_loader, optimizer=optimizer,
              epoch_index=epoch_index)

        w_acc, var_sim = valid(config=config, model=model,
                               dataloader=valid_loader, epoch_index=epoch_index)
        if not args.have_a_try and w_acc > best_w_acc:
            best_w_acc = w_acc
            config.logger.info(f'saving model {w_acc_best_model_path}')
            torch.save(model.state_dict(), w_acc_best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, choices=['java', 'javascript', 'python', 'c++'], required=True)
    parser.add_argument('--topk', type=int, required=True)

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--watermark_len', type=int, default=4)
    parser.add_argument('--var_selector_mask_probability', type=float, default='0.85')
    parser.add_argument('--substitute_mask_probability', type=float, default='-1')
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to load (default: all samples)")
    parser.add_argument('--have_a_try', action='store_true', help="try code, not train model")
    parser.add_argument('--config_path',
                        default='/home/liwei/Code-Watermark/variable-watermark/configs/my_model.yaml')
    args = parser.parse_args()
    main()
