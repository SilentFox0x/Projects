import torch
from torch.utils.data import Dataset
import json
from typing import Optional
from .watermark_factory import WatermarkFactory
import dgl
from tqdm import tqdm
from .bcb_dataset import get_var_info, build_graph, get_var_node_index_in_node_batch, VarReplaceHelper
import logging


class CsnDataset(Dataset):
    def __init__(self, filepath: str, temperature: float, use_type_or_text: str,
                 logger: Optional[logging.Logger] = None,
                 max_samples: Optional[int] = None):
        if logger is None:
            info = print
        else:
            info = logger.info

        if filepath.endswith('.json'):
            tensor_path = filepath.replace('.json', f'-tensor-{temperature:.1f}.pt')
        elif filepath.endswith('.jsonl'):
            tensor_path = filepath.replace('.jsonl', f'-tensor-{temperature:.1f}.pt')
        else:
            raise Exception(f'file {filepath} format error, should be json or jsonl')
        teacher_outputs = torch.load(tensor_path)
        if max_samples is not None:
            teacher_outputs = teacher_outputs[:max_samples]
        info(f'load teacher tensor: {tensor_path}, total {len(teacher_outputs)} samples')

        codes = []
        if filepath.endswith('.jsonl'):
            with open(file=filepath, mode='r', encoding='utf-8') as f:
                for line in tqdm(f.readlines(), desc='load code'):
                    codes.append(json.loads(line))
                    if max_samples is not None and len(codes) >= max_samples:
                        break
        else:
            with open(file=filepath, mode='r', encoding='utf-8') as f:
                codes = json.load(f)
                if max_samples is not None and len(codes) >= max_samples:
                    codes = codes[:max_samples]

        info(f'load code from {filepath}, total {len(codes)} samples')

        assert len(codes) == len(teacher_outputs), (
               Exception(f'codes length {len(codes)} != teacher_outputs length {len(teacher_outputs)}'))

        self.datapoints = []
        for index, code in enumerate(tqdm(codes)):
            var_infos = []
            try:
                assert len(code['subgraphs'].keys()) > 0, Exception(f'code {index} subgraphs is empty')
                for var, subgraph in code['subgraphs'].items():
                    assert var in teacher_outputs[index].keys(), (
                        f'teacher_outputs cannot find {var}')
                    assert teacher_outputs[index][var]['topk_idx'] is not None, (
                        f'teacher_outputs cannot find {var} topk_idx')

                    if use_type_or_text == 'text':
                        node_toks = [node[2] for node in subgraph['nodes']]  # text List[str]
                    elif use_type_or_text == 'type':
                        node_toks = [node[1] for node in subgraph['nodes']]  # type List[str]

                    assert len(node_toks) > 0, Exception(f'node_toks is 0')
                    var_info = {
                        'var': var,  # str
                        'node_toks': node_toks,
                        'edges': subgraph['edges'],  # List[List[start,end]]
                        'topk_idx': teacher_outputs[index][var]['topk_idx'],
                        'topk_prob': teacher_outputs[index][var]['topk_prob'],
                    }
                    var_infos.append(var_info)
            except Exception as e:
                info(e)
                continue
            self.datapoints.append({
                'function': code['function'],
                'var_sub_graphs': var_infos
            })

        self.length = len(self.datapoints)
        info(f'The dataset has {self.length} samples')

    def __getitem__(self, index):
        return self.datapoints[index]

    def __len__(self):
        return self.length


def csn_collate(datapoints, config, watermark_factory: WatermarkFactory):
    vars: list[str] = []
    var_tok_lens: list[int] = []
    var_tok_ids: list[list[int]] = []

    func_map_var_position = []  # 每个func的var数不确定，所以要记录func中的var在vars的begin-end关系
    var_position_in_node_batch = []  # 每个节点中var出现的begin-end位置
    graphs, watermarks, watermarks_class = [], [], []
    topk_idxs, topk_probs = [], []

    func_token_ids = []
    function_token_lens = []
    var_rename_pos_in_func_batch = []
    for func_index, datapoint in enumerate(datapoints):
        var_info_for_replace_var_in_function = {}
        func_map_var_position.append((len(vars), len(vars) + len(datapoint['var_sub_graphs'])))

        for var_info in datapoint['var_sub_graphs']:
            var = var_info['var']
            var_tok_len, var_tok_id = get_var_info(config, var)
            vars.append(var), var_tok_lens.append(var_tok_len), var_tok_ids.append(var_tok_id)

            var_index_in_vars = len(vars) - 1
            var_info_for_replace_var_in_function[var] = (var_index_in_vars, var_tok_len)

            var_position_in_node, graph = build_graph(config=config, var_index_in_vars=var_index_in_vars, var=var,
                                                      var_len=var_tok_len,
                                                      node_toks=var_info['node_toks'], edges=var_info['edges'])
            var_position_in_node_batch.extend(var_position_in_node)
            graphs.append(graph)

            if config.mode != 'test':
                topk_idxs.append(torch.tensor(var_info['topk_idx'], dtype=torch.long).to(config.device))
                topk_probs.append(torch.tensor(var_info['topk_prob'], dtype=torch.float).to(config.device))

        var_rename_pos_in_func, func_tok_ids, function_token_len = \
            locate_var_pos_in_function(config, datapoint['function'], var_info_for_replace_var_in_function)
        var_rename_pos_in_func_batch.append(var_rename_pos_in_func)
        func_token_ids.append(func_tok_ids)
        function_token_lens.append(function_token_len)

        w_bits, w_bits_class = watermark_factory.random_watermark()
        watermarks.append(w_bits), watermarks_class.append(w_bits_class)

    var_node_index_in_node_batch = get_var_node_index_in_node_batch(datapoints, sample_is_subgraph=False)
    var_node_index_in_node_batch = torch.tensor(var_node_index_in_node_batch, dtype=torch.long).to(config.device)

    graph_batch = dgl.batch(graphs).to(config.device)

    var_tok_ids = torch.tensor(var_tok_ids, dtype=torch.long)[:, :config.max_node_token_len]
    var_tok_ids = var_tok_ids.to(config.device)

    func_token_ids = torch.tensor(func_token_ids, dtype=torch.long).to(config.device)
    attention_masks = (func_token_ids != config.tokenizer.get_pad_id()).float().to(config.device)

    watermarks = torch.tensor(watermarks, dtype=torch.float).to(config.device)
    watermarks_class = torch.tensor(watermarks_class, dtype=torch.long).to(config.device)

    batch = {
        'vars': vars,
        'var_tok_lens': var_tok_lens,
        'var_tok_ids': var_tok_ids,
        'pre_var_tok_lens': var_tok_lens,
        'var_position_in_node_batch': var_position_in_node_batch,
        'var_node_index_in_node_batch': var_node_index_in_node_batch,
        'watermarks': watermarks,
        'watermarks_class': watermarks_class,
        'graph_batch': graph_batch,
        'ignore_n_subtoken': None,
        'topk_idxs': topk_idxs,
        'topk_probs': topk_probs,
        'func_token_ids': func_token_ids,
        'function_token_lens': function_token_lens,
        'attention_masks': attention_masks,
        'var_position_in_func_batch': var_rename_pos_in_func_batch,
        'func_map_var_position': func_map_var_position
    }
    return batch


def locate_var_pos_in_function(config, function: str, var_info_for_replace_function):
    func_tok_ids = []
    var_rename_pos_in_func = []
    for token in function.split(' '):
        if token == '':
            continue
        if token in var_info_for_replace_function.keys():
            var_index_in_vars, var_len = var_info_for_replace_function[token]
            var_rename_pos_in_func.append(
                VarReplaceHelper(var_index_in_vars, var_len, len(func_tok_ids)))
        func_tok_ids.extend(config.tokenizer.encode(token))

    func_tok_ids, function_token_len = pad_function_token_ids(config, func_tok_ids)
    var_rename_pos_in_func = [item for item in var_rename_pos_in_func if item.end < config.func_max_token_len]
    return var_rename_pos_in_func, func_tok_ids, function_token_len


def pad_function_token_ids(config, function_token_ids):
    function_token_ids = [config.tokenizer.bos_token_id] + function_token_ids[:config.func_max_token_len - 2] + [
        config.tokenizer.eos_token_id]
    function_token_len = len(function_token_ids)
    diff = config.func_max_token_len - function_token_len
    if diff > 0:
        function_token_ids.extend([config.tokenizer.pad_token_id] * diff)
    return function_token_ids, function_token_len
