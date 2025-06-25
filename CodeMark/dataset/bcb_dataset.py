import torch
from torch.utils.data import Dataset
import json
from typing import List, Tuple
from .watermark_factory import WatermarkFactory
import dgl
from itertools import chain
import torch.nn.functional as F
from tqdm import tqdm
import random


class MyDataset(Dataset):
    def __init__(self, config, filepath: str, name: str):
        self.name = name
        tensor_path = filepath.replace('.json', f'-tensor-{config.temperature:.1f}.pt')
        teacher_outputs = torch.load(tensor_path)
        config.logger.info(f'load teacher tensor: {tensor_path}')

        with open(file=filepath, mode='r', encoding='utf-8') as f:
            codes = json.load(f)
        self.datapoints = []
        for index, code in enumerate(tqdm(codes)):
            for var, subgraph in code['subgraphs'].items():
                if teacher_outputs[index][var]['topk_idx'] is None:
                    continue
                else:
                    if config.use_type_or_text == 'text':
                        node_toks = [node[2] for node in subgraph['nodes']]  # text List[str]
                    elif config.use_type_or_text == 'type':
                        node_toks = [node[1] for node in subgraph['nodes']]  # type List[str]
                    else:
                        raise KeyError
                    datapoint = {
                        'var': var,  # str
                        'node_toks': node_toks,
                        'edges': subgraph['edges'],  # List[List[start,end]]
                        'topk_idx': teacher_outputs[index][var]['topk_idx'],
                        'topk_prob': teacher_outputs[index][var]['topk_prob'],
                    }
                    self.datapoints.append(datapoint)
                if hasattr(config, 'only_first_var') and config.only_first_var is True:
                    break

        self.length = len(self.datapoints)
        config.logger.info('load ' + self.name + ' dataset from ' + filepath + ', total ' + str(self.length) + ' samples ')

    def __getitem__(self, index):
        return self.datapoints[index]

    def __len__(self):
        return self.length


def graph_collate(datapoints, config, watermark_factory: WatermarkFactory):
    """
    dgl会把多个graph压到一张大graph中, 所以每个graph的node数组会压到一个大的node数组中
    var_node_index_in_node_batch: 记录node数组中哪些node是var node
    var_position_in_node_batch: 记录node数组中 每个 node中var出现的位置, 方便在revealing中替换
    """
    vars, var_tok_lens, var_tok_ids = [], [], []
    # node_tok: 每个节点的subtoken_id, var_position_in_node_batch: 每个节点中var出现的begin-end位置
    var_position_in_node_batch = []
    graphs, watermarks, watermarks_class = [], [], []
    topk_idxs, topk_probs = [], []
    for datapoint in enumerate(datapoints):
        var = datapoint['var']
        var_tok_len, var_tok_id = get_var_info(config, var)
        vars.append(var), var_tok_lens.append(var_tok_len), var_tok_ids.append(var_tok_id)

        var_position_in_node, graph = build_graph(config=config, var_index_in_vars=len(vars)-1, var=var, var_len=var_tok_len,
                                                  node_toks=datapoint['node_toks'], edges=datapoint['edges'])
        var_position_in_node_batch.extend(var_position_in_node)
        graphs.append(graph)

        w_bits, w_bits_class = watermark_factory.random_watermark()
        watermarks.append(w_bits), watermarks_class.append(w_bits_class)

        if config.mode != 'test':
            topk_idxs.append(torch.tensor(datapoint['topk_idx'], dtype=torch.long).to(config.device))
            topk_probs.append(torch.tensor(datapoint['topk_prob'], dtype=torch.float).to(config.device))

    watermarks = torch.tensor(watermarks, dtype=torch.float).to(config.device)
    watermarks_class = torch.tensor(watermarks_class, dtype=torch.long).to(config.device)
    var_node_index_in_node_batch = get_var_node_index_in_node_batch(datapoints)
    var_node_index_in_node_batch = torch.tensor(var_node_index_in_node_batch, dtype=torch.long).to(config.device)
    graph_batch = dgl.batch(graphs).to(config.device)

    max_subtoken_len = max(var_tok_lens)
    var_tok_ids = torch.tensor(var_tok_ids, dtype=torch.long)[:, :max_subtoken_len]
    var_tok_ids = var_tok_ids.to(config.device)

    pre_var_tok_lens = [var_len for var_len in var_tok_lens]
    ignore_n_subtoken = [0 for _ in var_tok_lens]  # 1/2的概率ignore 第0个subtoken

    batch = {
        'vars': vars,
        'var_tok_lens': var_tok_lens,
        'var_tok_ids': var_tok_ids,
        'pre_var_tok_lens': pre_var_tok_lens,
        'var_position_in_node_batch': var_position_in_node_batch,
        'var_node_index_in_node_batch': var_node_index_in_node_batch,
        'watermarks': watermarks,
        'watermarks_class': watermarks_class,
        'graph_batch': graph_batch,
        'ignore_n_subtoken': ignore_n_subtoken,
        'topk_idxs': topk_idxs,
        'topk_probs': topk_probs,
    }
    return batch


def get_var_info(config, var: str) -> Tuple[int, List[int]]:
    var_tok_id = config.tokenizer.encode(var)
    var_tok_id, var_tok_len = pad_truncature_tok_id(tok_ids=var_tok_id,
                                                    max_len=config.max_node_token_len,
                                                    pad_id=config.tokenizer.get_pad_id())
    return var_tok_len, var_tok_id


class VarReplaceHelper:
    def __init__(self, var_index_in_vars: int, var_len: int, begin: int):
        self.var_index_in_vars = var_index_in_vars
        self.var_len = var_len
        self.begin = begin
        self.end = begin + var_len


def build_graph(config, var_index_in_vars: int, var: str, var_len: int, node_toks: List[str], edges, mask_set=None):
    edges.append((0, 0))
    src = [x[0] for x in edges]
    tgt = [x[1] for x in edges]
    graph = dgl.graph((src, tgt))
    graph = dgl.add_reverse_edges(graph)

    node_tok_ids, node_tok_lens, var_position_in_node = [], [], []
    for text in node_toks:
        ids, var_position = [], []
        text = [item for item in text.split(' ') if item != '']
        for token in text:
            if token == var:
                var_position.append(VarReplaceHelper(var_index_in_vars, var_len, len(ids)))
            if mask_set is not None and token in mask_set:
                token = 'MASK'
            # if token != var:  # hiding中输入的subgraph里删去var, 看只用上下文能不能向teacher probs靠拢
            #     ids.extend(config.tokenizer.encode(token))
            # else:
            #     for _ in range(var_len):
            #         ids.extend(config.tokenizer.encode(' '))
            ids.extend(config.tokenizer.encode(token))

        ids, id_len = pad_truncature_tok_id(ids, config.max_node_token_len, config.tokenizer.get_pad_id())
        node_tok_ids.append(ids), node_tok_lens.append(id_len)

        var_position = [item for item in var_position if item.end < config.max_node_token_len]
        var_position_in_node.append(var_position)
    graph.ndata['node_tok_ids'] = torch.tensor(node_tok_ids, dtype=torch.long)
    graph.ndata['node_tok_lens'] = torch.tensor(node_tok_lens, dtype=torch.long)
    return var_position_in_node, graph


class BatchStatementCodeGPTInput:
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


def get_var_statements_info(config, statements: List[str],
                            var: str, var_index_in_vars: int, var_len: int) -> \
        Tuple[List[List[int]], List[List[VarReplaceHelper]]]:
    if len(statements) > config.max_statements:
        statements = statements[:config.max_statements]
    else:
        statements.extend(['' for _ in range(config.max_statements - len(statements))])

    tok_ids_in_statement, var_position_in_statement = [], []
    for statement in statements:
        ids, var_position = [], []
        for token in statement.split(' '):
            if token == var:
                var_position.append(VarReplaceHelper(var_index_in_vars, var_len, len(ids)))
            ids.extend(config.tokenizer.encode(token))

        ids, _ = pad_truncature_tok_id(ids, config.max_statement_tokens, config.tokenizer.pad_token_id)
        tok_ids_in_statement.append(ids)

        var_position = [item for item in var_position if item.end < config.max_statement_tokens]
        var_position_in_statement.append(var_position)

    return tok_ids_in_statement, var_position_in_statement


def get_batch_statement_input(config, tok_ids_in_statement_batch: List[List[List[int]]],
                              var_position_in_statement_batch) -> Tuple[BatchStatementCodeGPTInput, List[List[int]]]:
    input_ids = list(chain.from_iterable(tok_ids_in_statement_batch))
    input_ids = torch.stack([torch.LongTensor(x) for x in input_ids]).to(config.device)
    attention_mask = (input_ids != config.tokenizer.pad_token_id).float().to(config.device)
    labels = input_ids.masked_fill(input_ids == config.tokenizer.pad_token_id, -100).to(config.device)

    var_position_in_statement_batch = list(chain.from_iterable(var_position_in_statement_batch))

    return BatchStatementCodeGPTInput(input_ids, attention_mask, labels), var_position_in_statement_batch


def get_var_node_index_in_node_batch(datapoints, sample_is_subgraph=True):
    # dgl.batch 会把每个变量的图拼成一个大图, 需要记录变量节点在大图中对应的位置
    # sample_is_subgraph True 时表示每个样本是var-subgraph，False 时是每个样本是一个func
    var_node_index_in_node_batch = [0]

    for datapoint in datapoints:
        if sample_is_subgraph:
            node_num = len(datapoint['node_toks'])
            var_node_index_in_node_batch.append(var_node_index_in_node_batch[-1] + node_num)
        else:
            for var_info in datapoint['var_sub_graphs']:
                node_num = len(var_info['node_toks'])
                var_node_index_in_node_batch.append(var_node_index_in_node_batch[-1] + node_num)
    var_node_index_in_node_batch = var_node_index_in_node_batch[:-1]
    return var_node_index_in_node_batch


def pad_truncature_tok_id(tok_ids: List[int], max_len: int, pad_id: int) -> Tuple[List[int], int]:
    tok_ids_len = len(tok_ids)
    if tok_ids_len == 0:
        tok_ids.append(pad_id)
        tok_ids_len = 1
    if tok_ids_len >= max_len:
        tok_ids = tok_ids[:max_len]
        tok_ids_len = max_len
    else:
        diff = max_len - tok_ids_len
        tok_ids.extend([pad_id] * diff)
    return tok_ids, tok_ids_len


def norm_teacher_tensor(teacher_output: torch.Tensor, t_norm_way: str) -> torch.Tensor:
    if t_norm_way == 'ignore':
        return teacher_output
    elif t_norm_way == 'L2':
        teacher_output_norm = torch.norm(teacher_output, dim=1, keepdim=True)
        if torch.is_nonzero(torch.sum(teacher_output_norm)):
            return teacher_output / teacher_output_norm
        else:
            return teacher_output
    elif t_norm_way == 'softmax':
        return F.softmax(teacher_output, dim=-1)
    else:
        raise Exception('incorrect norm way')
