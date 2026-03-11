import argparse
from tqdm import tqdm
import json
import time
import os
import traceback
from typing import List

import torch
from dataset import WatermarkFactory, csn_collate
from model import CodeMark
from configs import Config
from preprocess import MyParser
from utils import SimpleBefore, replace_var

no_subgraph = 0
exception_num = 0


def read_test_data(test_data_path) -> List:
    test_datasets = []
    with open(test_data_path, mode='r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            # code_tokens = line['code_tokens']
            # my_original_string = ''
            # for token in code_tokens:
            #     if token in [';', '{', '}']:
            #         token += '\n'
            #     my_original_string += token + ' '
            simple_before = SimpleBefore(docstring_tokens=line['docstring_tokens'],
                                         original_string=line['original_string'],
                                         # original_string=my_original_string,
                                         )
            test_datasets.append(simple_before)
            if args.max_samples is not None and i >= args.max_samples:
                break

    print(f'load test dataset from {test_data_path}')
    return test_datasets


def embed(config, watermark_factory, model, my_parser, simple_before):
    function = simple_before.original_string
    var_map = {}
    watermarks = []
    try:
        var_infos = []
        subgraphs = my_parser.parse_code(function)
        for var, subgraph in subgraphs.items():
            node_toks = [node[2] for node in subgraph['nodes']]  # [0, 'identifier', 'a'] -> ['a']
            var_info = {
                'var': var,  # str
                'node_toks': node_toks,
                'edges': subgraph['edges'],  # List[List[start,end]]
            }
            var_infos.append(var_info)
        if len(var_infos) == 0:
            raise Exception('no subgraph')
        datapoints = [{
            'function': simple_before.original_string,
            'var_sub_graphs': var_infos
        }]

        with torch.no_grad():
            batch = csn_collate(datapoints, config, watermark_factory)

            var_map = model.embed(batch)

            for bits in batch['watermarks'].cpu().detach().tolist():
                watermarks = list(map(int, bits))
    except Exception as e:
        global exception_num
        exception_num += 1
        traceback.print_exc()
        return None

    after_watermark = replace_var(simple_before.original_string, var_map)

    if args.print_pre_var:
        print(var_map)

    simple_after = {
        "docstring_tokens": simple_before.docstring_tokens,
        "output_origin_func": False,
        "original_string": simple_before.original_string,
        "after_watermark": after_watermark,
        "watermark": watermarks,
        "extract": []
    }
    return simple_after


def main():
    config = Config(args.config_path, mode='test')
    config.language = args.lang
    config.device = 'cuda:' + args.device
    config.watermark_len = args.watermark_len
    config.VarSelector['mask_probability'] = args.var_selector_mask_probability
    config.VarDecoder['substitute_mask_probability'] = args.substitute_mask_probability
    config.VarDecoder['topk'] = args.topk

    model = CodeMark(config)
    model_name = args.model_name
    save_model_path = os.path.join(config.model_save_path, f'{model_name}.pth')
    model.load_state_dict(torch.load(save_model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()

    watermark_factory = WatermarkFactory(config.watermark_len)

    test_datasets = read_test_data(config.dataset_path[config.language]['test_data_path'])

    start_time = time.time()

    simples = []
    my_parser = MyParser(args.lang)
    for index, simple_before in enumerate(tqdm(test_datasets)):
        simple_after = embed(config=config, watermark_factory=watermark_factory,
                             model=model, my_parser=my_parser, simple_before=simple_before)
        if simple_after is not None:
            simples.append(json.dumps(simple_after))

    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_datasets)

    with open(args.output_jsonl_path, mode='w', encoding='utf-8') as f:
        for simple in simples:
            f.write(simple + "\n")
    print(f'save embed result to {args.output_jsonl_path}')

    global no_subgraph
    print(f'{no_subgraph} has no subgraph, {exception_num} exceptions, each sample use {avg_time:4f}')


def temp_test_before_and_after_watermark_func_has_same_var_num():
    my_parser = MyParser('java')
    after_wm = '/home/liwei/Code-Watermark/variable-watermark/results/embed/11-17-16-31-45-E3-w-acc-best-embed.jsonl'

    with open(after_wm, mode='r', encoding='utf-8') as f:
        after_lines = f.readlines()

    for i in range(len(after_lines)):
        if i < 100 or i > 200:
            continue
        print(i)
        obj = json.loads(after_lines[i])
        b_subgraphs = my_parser.parse_code(obj['original_string'])
        a_subgraphs = my_parser.parse_code(obj['after_watermark'])
        print(b_subgraphs.keys())
        print(a_subgraphs.keys())
        print('===== ori =====')
        print(obj['original_string'])
        print('watermarked:')
        print(obj['after_watermark'])


def temp_test_empty_node_token():
    func = '''
function FormField ( ) { 
  const controlProps  = 1
    return (
      < ElementType  className={ classes }> 
        < label > 
          { createElement ( control , controlProps )} { label } // jsx_text
        </ label > 
      </ ElementType > 
    )

}
    '''
    config = Config(args.config_path, mode='test')
    config.device = 'cuda:' + args.device
    my_parser = MyParser(args.lang)
    watermark_factory = WatermarkFactory(config.watermark_len)
    var_infos = []
    subgraphs = my_parser.parse_code(func)
    for var, subgraph in subgraphs.items():
        node_toks = [node[2] for node in subgraph['nodes']]
        var_info = {
            'var': var,  # str
            'node_toks': node_toks,
            'edges': subgraph['edges'],  # List[List[start,end]]
        }
        var_infos.append(var_info)
    if len(var_infos) == 0:
        raise Exception('no subgraph')
    datapoints = [{
        'function': func,
        'var_sub_graphs': var_infos
    }]

    csn_collate(datapoints, config, watermark_factory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_jsonl_path', type=str, required=True, help="path to output")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--topk', type=int, required=True)

    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to load (default: all samples)")
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--watermark_len', type=int, default=4)
    parser.add_argument('--var_selector_mask_probability', type=float, default=0.85)
    parser.add_argument('--substitute_mask_probability', type=float, default='-1')
    parser.add_argument('--print_pre_var', action='store_true')
    parser.add_argument('--config_path',
                        default='/home/liwei/Code-Watermark/variable-watermark/configs/my_model.yaml')
    args = parser.parse_args()
    main()
