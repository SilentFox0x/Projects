import json
import os
import sys
from tqdm import tqdm
import re

DATASET = '../CALS/test.jsonl'

def NL_split(input):
    pat = re.compile(r"([.()!<>{};,@\[\]\"\'_])")
    input = pat.sub(" \\1 ", input)
    return input.split()

def NL_unsplit(input):
    outstr=''
    for x in input:
        outstr+=(x+' ')
    return outstr[:-1]

split=NL_split
unsplit=NL_unsplit

def delete_comment(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)

def reverse(result_file_path:str):
    flag = False
    output = open(result_file_path[:-6]+'_reverse.jsonl', 'w', encoding="utf-8")
    dataset = []
    with open(DATASET, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dataset.append(json.loads(line)['original_string'])
    f.close()
    i = -1
    with open(result_file_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            data = eval(line)
            flag = data['if_watermarked']
            i+=1
            out = data
            out['original_string'] = dataset[i]
            if not flag:
                out['after_watermark'] = dataset[i]
            else:
                # k+=1
                # num_token_oristr = len(split(delete_comment(dataset[i])))
                # num_after_watermark = len(split(data['after_watermark']))
                # if num_token_oristr < num_after_watermark:
                # print(split(delete_comment(dataset[i])))
                # print(split(data['after_watermark']))
                # exit(1)

                ret = recover_modified_code(delete_comment(dataset[i]), split(delete_comment(dataset[i])), split(data['after_watermark']))
                out['after_watermark'] = ret
                # out['after_watermark_reverse'] = reverse_NL_split_one_space(out['after_watermark'])
            output.writelines(str(out)+'\n')    
    f.close()
    # print(j,'  ',k)

def reverse_NL_split_one_space(input):
    pat = re.compile(r"\s*([.()!<>{};,@\[\]])\s*")
    reversed_input = pat.sub(r"\1", input)
    reversed_input = re.sub(r"\s+", " ", reversed_input)
    return reversed_input

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp


def recover_modified_code(code_str_A, token_list_B, modified_token_list_C):
    def find_next_token(start_index, token, code_str):
        index = code_str.find(token, start_index)
        while index >= 0 and code_str[index - 1:index + len(token) + 1] in token_list_B:
            index = code_str.find(token, index + 1)
        return index
    tokens_mapping = []
    n, m = len(token_list_B), len(modified_token_list_C)
    dp = edit_distance(token_list_B, modified_token_list_C)

    i, j = n, m
    while i > 0 and j > 0:
        min_edit_dis = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        if min_edit_dis == dp[i - 1][j - 1]:
            tokens_mapping.append((token_list_B[i - 1], modified_token_list_C[j - 1]))
            i -= 1
            j -= 1
        elif min_edit_dis == dp[i - 1][j]:
            tokens_mapping.append((token_list_B[i - 1], None))
            i -= 1
        else:
            tokens_mapping.append((None, modified_token_list_C[j - 1]))
            j -= 1

    while i > 0:
        tokens_mapping.append((token_list_B[i - 1], None))
        i -= 1

    while j > 0:
        tokens_mapping.append((None, modified_token_list_C[j - 1]))
        j -= 1

    tokens_mapping = tokens_mapping[::-1]
    code_str_A_index = 0
    modified_code = ""

    for original_token, modified_token in tokens_mapping:
        if original_token is not None:
            token_start_index = find_next_token(code_str_A_index, original_token, code_str_A)
            modified_code += code_str_A[code_str_A_index:token_start_index]
            code_str_A_index = token_start_index

        if modified_token is not None:
            modified_code += modified_token

        if original_token is not None:
            code_str_A_index += len(original_token)

    modified_code += code_str_A[code_str_A_index:]

    return modified_code


if __name__ == '__main__':
    reverse(sys.argv[1])

