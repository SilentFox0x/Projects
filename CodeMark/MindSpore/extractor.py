import argparse
from dataset import WatermarkFactory, csn_collate
from model import CodeMark
import torch
from configs import Config
from tqdm import tqdm
from preprocess import MyParser
import time
import os
from utils import SampleAfter, read_embed_data
import traceback

no_subgraph = 0
exception_num = 0


def extract_from_sample(config, watermark_factory, model, my_parser, sample_after: SampleAfter):
    function = sample_after.after_watermark
    try:
        var_infos = []
        subgraphs = my_parser.parse_code(function)
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
            'function': sample_after.after_watermark,
            'var_sub_graphs': var_infos
        }]
        with torch.no_grad():
            batch = csn_collate(datapoints, config, watermark_factory)
            extract_watermarks = model.extract(batch)
    except Exception as e:
        global exception_num
        exception_num += 1
        traceback.print_exc()
        return None

    sample_after.extract = extract_watermarks[0]
    return sample_after


def main():
    config = Config(args.config_path, mode='test')
    config.language = args.lang
    config.device = 'cuda:' + args.device
    config.watermark_len = args.watermark_len

    model = CodeMark(config)
    model_name = args.model_name
    save_model_path = os.path.join(config.model_save_path, f'{model_name}.pth')
    model.load_state_dict(torch.load(save_model_path, map_location= config.device))
    model = model.to(config.device)
    model.eval()

    watermark_factory = WatermarkFactory(config.watermark_len)

    embed_datasets = read_embed_data(args.input_jsonl_file, args.max_samples)

    start_time = time.time()

    samples = []
    my_parser = MyParser(args.lang)
    for index, sample_before in enumerate(tqdm(embed_datasets)):
        obj = extract_from_sample(config=config, watermark_factory=watermark_factory,
                                  model=model, my_parser=my_parser, sample_after=sample_before)
        if obj is not None:
            samples.append(obj.toJSON())

    end_time = time.time()
    avg_time = (end_time - start_time) / len(embed_datasets)

    with open(args.output_jsonl_path, mode='w', encoding='utf-8') as f:
        for simple in samples:
            f.write(simple + "\n")
    print(f'save extract result to {args.output_jsonl_path}')
    global no_subgraph
    print(f'{no_subgraph} has no subgraph, {exception_num} exceptions, each sample use {avg_time:4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_jsonl_file', type=str, required=True, help="path to watermarked code")
    parser.add_argument('-o', '--output_jsonl_path', type=str, required=True, help="path to extracted watermark")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)

    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to load (default: all samples)")
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--watermark_len', type=int, default=4)
    parser.add_argument('--config_path',
                        default='/home/liwei/Code-Watermark/variable-watermark/configs/my_model.yaml')
    args = parser.parse_args()
    main()

# --model_name   --embed_path embed/placeholder-embed.jsonl --extract_path extract/placeholder-extract.jsonl

# attack
# --model_name 04-28-12-25-19-E3 --embed_path /home/borui/code-watermarking/data/temp_1_csn_funcs.jsonl --extract_path /home/liwei/Code-Watermark/variable-watermark/results/extract_from_attack/04-28-12-25-19-E3-temp1-extract.jsonl
# /home/liwei/miniconda3/envs/variable_watermark/bin/python3 /home/liwei/Code-Watermark/variable-watermark/extractor.py --model_name 04-28-12-25-19-E3 --embed_path /home/borui/code-watermarking/data/temp_2_csn_funcs.jsonl --extract_path /home/liwei/Code-Watermark/variable-watermark/results/extract_from_attack/04-28-12-25-19-E3-temp2-extract.jsonl