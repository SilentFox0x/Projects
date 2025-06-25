from collections import defaultdict
import random
from typing import List, Optional, Dict
import json
from tree_sitter import Language, Parser

from .grammar_checker import is_valid


class SimpleBefore:
    def __init__(self, docstring_tokens, original_string):
        self.original_string = original_string
        self.docstring_tokens = docstring_tokens
        self.watermark = None

    def set_watermark(self, watermark):
        self.watermark = watermark


class SampleAfter:
    def __init__(self, docstring_tokens, output_origin_func: bool,
                 after_watermark: str, original_string: str, watermark: List[int]):
        self.docstring_tokens = docstring_tokens
        self.output_origin_func = output_origin_func
        self.original_string = original_string
        self.after_watermark = after_watermark
        self.watermark = watermark
        self.extract = []

    def set_extract_watermark(self, watemarks):
        self.extract = watemarks

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


def read_embed_data(embed_path: str, max_samples: Optional[int] = None) -> List[SampleAfter]:
    print(f'load embed data from {embed_path}')
    embed_data = []
    with open(embed_path, mode='r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            sample = SampleAfter(
                docstring_tokens=line['docstring_tokens'],
                original_string=line['original_string'],
                output_origin_func=line['output_origin_func'],
                after_watermark=line['after_watermark'],
                watermark=line['watermark'],
            )
            embed_data.append(sample)
            if max_samples is not None and i >= max_samples:
                break
    return embed_data


def replace_var(func: str, var_pre_var_map: dict) -> str:
    code_tokens: List[str] = func.split(' ')
    code_tokens = [item for item in code_tokens if item != '']
    watermarked_code_tokens = []
    for token in code_tokens:
        pre_var = var_pre_var_map.get(token.strip(), None)
        if pre_var is not None:
            watermarked_code_tokens.append(pre_var)
        else:
            watermarked_code_tokens.append(token)

    after_watermark = ' '.join(watermarked_code_tokens)
    return after_watermark


def replace_var_by_ast(tree, var_pre_var_map: dict) -> str:
    # 这个方法不行，得到的func string没有换行
    root = tree.root_node

    def _ast_to_text(node):
        if len(node.children) > 0:
            text = ' '
            for child in node.children:
                text += ' ' + _ast_to_text(child)
        else:
            text = node.text.decode("utf-8")
        return text
    text = _ast_to_text(root)
    return text

####################
# name convention
def detect_naming_style(variable_name):
    if '_' in variable_name:
        if variable_name.isupper():
            return 'screaming_snake_case'
        else:
            return 'snake_case'
    else:
        if not variable_name:
            return 'unknown'
        first_char = variable_name[0]
        remaining = variable_name[1:]
        if first_char.isupper():
            return 'pascal_case'
        else:
            has_upper = any(c.isupper() for c in remaining)
            return 'camel_case' if has_upper else 'camel_case'

def convert_subtokens(style, subtokens):
    if not subtokens:
        return ""
    if style == 'snake_case':
        return '_'.join([token.lower() for token in subtokens])
    elif style == 'screaming_snake_case':
        return '_'.join([token.upper() for token in subtokens])
    elif style == 'camel_case':
        camel_tokens = [subtokens[0].lower()]
        for token in subtokens[1:]:
            camel_tokens.append(token.capitalize())
        return ''.join(camel_tokens)
    elif style == 'pascal_case':
        return ''.join([token.capitalize() for token in subtokens])
    else:
        return '_'.join([token.lower() for token in subtokens])

def follow_name_convention(ori_vars: List[str], sub_toks: List[List[str]]) -> List[str]:
    pre_vars = []
    for i in range(len(ori_vars)):
        ori_var = ori_vars[i]
        subtok = sub_toks[i]
        style = detect_naming_style(ori_var)
        pre_var = convert_subtokens(style, subtok)
        pre_vars.append(pre_var)
    return pre_vars
####################


def mask_invalid_var(pre_vars: List[str]) -> None:
    for i in range(len(pre_vars)):
        if not is_valid(pre_vars[i]):
            pre_vars[i] = 'UN-VALID'


def mask_duplicate_var(pre_vars: List[str]) -> None:
    var_to_index = defaultdict(list)
    for index, pre_var in enumerate(pre_vars):
        var_to_index[pre_var].append(index)
    for pre_var, indexes in var_to_index.items():
        if len(indexes) > 1:
            for index in indexes:
                pre_vars[index] = 'UN-VALID'
            pre_vars[random.choice(indexes)] = pre_var


def get_var_map(ori_vars: List[str], sub_toks: List[List[str]]) -> Dict[str, str]:
    pre_vars = follow_name_convention(ori_vars, sub_toks)
    mask_invalid_var(pre_vars)
    mask_duplicate_var(pre_vars)

    var_map = {}
    for var, pre_var in zip(ori_vars, pre_vars):
        if pre_var != 'UN-VALID':
            var_map[var] = pre_var
    return var_map


def get_var_from_var_id(config, var_tok_lens: List[int], var_ids) -> List[List[str]]:
    sub_toks = []
    for i in range(len(var_tok_lens)):
        pre_var_len = var_tok_lens[i]
        temp = var_ids[i]
        temp = temp[:pre_var_len]
        temp = temp.cpu().tolist()
        sub_tok = [config.tokenizer.decode(item).strip() for item in temp]
        sub_toks.append(sub_tok)
    return sub_toks


if __name__ == '__main__':
    language = Language('/home/liwei/Code-Watermark/variable-watermark/resources/my-languages.so', 'javascript')
    parser = Parser()
    parser.set_language(language)
    code = '''
    async function get(action, { getService } = {}) {
        const {
            service: serviceId = null,
            onlyMappedValues = false,
            endpoint
        } = action.payload
    }
    '''
    tree = parser.parse(bytes(code, 'utf8'))
    var_pre_var_map = {'endpoint': 'my_endpoint'}
    new_code = replace_var_by_ast(tree, var_pre_var_map)
    print(new_code)

    new_code = replace_var(code, var_pre_var_map)
    print(new_code)
