import argparse
from typing import List, Tuple
import json
from tqdm import tqdm
import os

import torch
from transformers import RobertaTokenizer

from preprocess import MyParser
from distill_knowledge import TeacherModel, _tokenize
from preprocess.data_flow.utils import remove_comments_and_docstrings
from preprocess.prepare_helper import Spliter


class FunFilter:
    def __init__(self, lang: str):
        assert lang in ['java', 'javascript', 'python', 'c++']
        self.lang = lang
        self.spliter = Spliter(lang=lang)
        self.my_parser = MyParser(lang)
        code_bert_path = 'microsoft/codebert-base'
        self.tokenizer_mlm = RobertaTokenizer.from_pretrained(code_bert_path, local_files_only=True)

    def check_func(self, func: str):
        if self.lang == 'java':
            func = 'public class A{ \n' + func + ' } \n'
        else:
            func = func

        # delete more than 500 lines
        if len(func.split('\n')) > 500:
            raise Exception(' more than 500 lines')

        # delete comments
        func = remove_comments_and_docstrings(source=func, lang=self.lang)

        has_error, tree = self.my_parser.ast_has_error(func)
        if has_error:
            raise Exception('ast has error')

        # delete no var
        subgraphs = self.my_parser.parse_code(func, tree)
        if len(subgraphs) == 0:
            raise Exception('no var')
        if len(subgraphs) > 25:
            raise Exception('too many var')

        # add space_to_identifier
        func = self.spliter.add_space_to_identifier(func)

        # delete token_num > 510
        tokens, sub_tokens, occurrences = _tokenize(func, self.tokenizer_mlm)
        if len(sub_tokens) >= 510:
            raise Exception('token num > 510')

        return func, subgraphs


def load_jsonl_data(dir_path) -> List[str]:
    data = []
    for file in os.listdir(dir_path):
        if file.endswith('jsonl'):
            with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:
                data.extend(f.readlines())

    if args.max_samples is not None:
        data = data[:args.max_samples]
    return data


def process_filter(func_filter, raw_data: List[str], is_test: bool) -> List[str]:
    results = []
    for item in tqdm(raw_data, desc='filter raw data'):
        obj = json.loads(item)
        func = obj['original_string']
        try:
            func, subgraphs = func_filter.check_func(func)
        except Exception as e:
            if args.verbose:
                print(e)
            continue

        if is_test:
            obj['original_string'] = func
            results.append(json.dumps(obj))
        else:
            sample = {
                'function': func,
                'subgraphs': subgraphs,
            }
            results.append(json.dumps(sample))
    return results


def get_teacher_substitutions(teacher, processed_data: List[str]) -> Tuple[List, List[str]]:
    substitutions, processed_data_with_substitutions = [], []
    for obj in tqdm(processed_data, desc='get teacher substitutions'):
        code = json.loads(obj)
        func = code['function']
        variables = code['subgraphs'].keys()

        variable_substitution = teacher.get(func, variables)

        all_var_has_substitutions = True
        for var in variables:
            if var not in variable_substitution.keys() or variable_substitution[var]['topk_idx'] is None:
                all_var_has_substitutions = False
                break

        if all_var_has_substitutions:
            processed_data_with_substitutions.append(obj.strip())
            substitutions.append(variable_substitution)

    return substitutions, processed_data_with_substitutions


def main():
    func_filter = FunFilter(args.lang)
    for part in ['train', 'valid', 'test']:
        print(f'processing {part} data')

        raw_data_dir = os.path.join(args.raw_data_dir, f'{args.lang}', 'final/jsonl', f'{part}')
        raw_data = load_jsonl_data(raw_data_dir)
        print(f'total raw data is {len(raw_data)}')

        data = process_filter(func_filter, raw_data, part == 'test')
        print(f'after filter, total data is {len(data)}')

        processed_data_path = os.path.join(args.processed_data_dir, f'{args.lang}', f'{part}-tmp.jsonl')
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        with open(processed_data_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item + '\n')
        print(f'save processed_data to {processed_data_path}')

    teacher = TeacherModel(args.temperature, args.device)
    for part in ['train', 'valid']:
        processed_data_path = os.path.join(args.processed_data_dir, f'{args.lang}', f'{part}-tmp.jsonl')
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                data.append(line)
                if args.max_samples is not None and len(data) >= args.max_samples:
                    break

        print(f"total {part} processed_data is {len(data)}")
        substitutions, data = get_teacher_substitutions(teacher, data)

        processed_data_path = os.path.join(args.processed_data_dir, f'{args.lang}', f'{part}.jsonl')
        with open(processed_data_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item + '\n')
        print(f'after filter no substitutions, total data is {len(data)}, save to {processed_data_path}')

        save_path = processed_data_path.replace('.jsonl', f'-tensor-{args.temperature:.1f}.pt')
        torch.save(substitutions, save_path)
        print(f'save teacher_output to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, choices=['java', 'javascript', 'python', 'c++'], required=True)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('-r', '--raw_data_dir', type=str, required=True)
    parser.add_argument('-p', '--processed_data_dir', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to load (default: all samples)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--config_path',
                        default='/home/liwei/Code-Watermark/variable-watermark/configs/my_model.yaml')

    args = parser.parse_args()
    main()
