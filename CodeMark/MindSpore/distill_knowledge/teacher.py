import torch
import torch.nn.functional as F
from typing import List
from transformers import RobertaForMaskedLM, RobertaTokenizer
from .python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from .tutil import _tokenize, get_variable_posistions_from_code, TokenBeginEndInSubToken
from utils import is_valid


class TeacherModel:
    def __init__(self, temperature, device: str):
        code_bert_path = 'microsoft/codebert-base'
        self.model_mlm = RobertaForMaskedLM.from_pretrained(code_bert_path, local_files_only=True)
        self.tokenizer_mlm = RobertaTokenizer.from_pretrained(code_bert_path, local_files_only=True)
        self.device = f'cuda:{device}'
        self.model_mlm.to(self.device)
        self.block_size = 512
        self.temperature = temperature

        # self.illegal_vars = {}
        # self._set_illegal_vars()

    def _set_illegal_vars(self):
        vocab = self.tokenizer_mlm.get_vocab()
        for idx in range(len(vocab.items())):
            var = self.tokenizer_mlm.decode(idx)
            var = var.strip()
            if not is_valid(var):
                self.illegal_vars[idx] = var
        print(f'total {len(self.illegal_vars.items())} illegal vars')

    def get(self, func: str, pre_define_vars: List[str]):
        identifiers, code_tokens = get_identifiers(func)

        processed_code = " ".join(code_tokens)

        tokens, sub_tokens, occurrences = _tokenize(processed_code, self.tokenizer_mlm)
        variable_names = []
        # for i, name in enumerate(identifiers): # identifiers: [[a],[b]]
        #     if ' ' in name[0].strip():
        #         continue
        #     variable_names.append(name[0])
        variable_names = pre_define_vars

        sub_tokens = [self.tokenizer_mlm.cls_token] + sub_tokens[:self.block_size - 2] + [self.tokenizer_mlm.sep_token]
        input_ids_ = torch.tensor([self.tokenizer_mlm.convert_tokens_to_ids(sub_tokens)]).to(self.device)

        with torch.no_grad():
            model_predictions = self.model_mlm(input_ids_)[0].squeeze()  # len(sub_tokens), vocab
            model_predictions = model_predictions[1:-1, :]  # remove cls and sep

            names_positions_dict = get_variable_posistions_from_code(tokens, variable_names)
            variable_substitution = {}
            for variable, positions in names_positions_dict.items():
                topk_prob, topk_idx = self.get_all_substitues(positions, occurrences, model_predictions)
                variable_substitution[variable] = {"topk_prob": topk_prob, "topk_idx": topk_idx}
        return variable_substitution

    def get_all_substitues(self, positions, occurrences, model_predictions):
        all_substitues = []
        subtoken_lens = None
        for token_index in positions:
            occu = occurrences[token_index]
            if occu.end >= model_predictions.size()[0]:
                break
            substitutes = model_predictions[occu.begin:occu.end]
            # [sub_token_len, vocab_size]
            substitutes = F.softmax(substitutes / self.temperature, dim=-1)
            # [sub_token_len x vocab_size,]
            substitutes = substitutes.reshape(-1)
            all_substitues.append(substitutes)

        if len(all_substitues) == 0:
            # print('cannot find substitutions')
            # all_substitues.append(torch.zeros(size=[subtoken_lens, self.tokenizer_mlm.vocab_size], dtype=torch.float))
            return None, None
        else:
            # [sub_token_len x vocab_size, ...] -> [Occurence, sub_token_len x vocab_size]
            all_substitues = torch.stack(all_substitues, dim=0)
            all_substitues = torch.sum(all_substitues, dim=0)  # [sub_token_len x vocab_size]
            all_substitues = all_substitues.reshape(-1, self.tokenizer_mlm.vocab_size)  # [sub_token_len, vocab_size]
            # all_substitues = F.softmax(all_substitues, dim=-1)
            # all_substitues = all_substitues / torch.norm(all_substitues, dim=1, keepdim=True)

            topk_prob, topk_idx = torch.topk(all_substitues, k=10, dim=-1)
            topk_prob = F.softmax(topk_prob, -1)
            topk_prob = topk_prob.detach().cpu().tolist()
            topk_idx = topk_idx.detach().cpu().tolist()
            # topk_prob, topk_idx = self.delete_illegal_and_duplicate(all_substitues, k=10)

            # self.show_top_k(topk_idx, topk_prob)
            return topk_prob, topk_idx

    def delete_illegal_and_duplicate(self, all_substitues, k: int):
        topk_prob, topk_idx = torch.topk(all_substitues, k=1000, dim=-1)
        topk_prob = topk_prob.detach().cpu().tolist()
        topk_idx = topk_idx.detach().cpu().tolist()

        selected_idxes, selected_probs = [], []
        selected_vars = []
        # 去除不合法,
        for idx, prob in zip(topk_idx, topk_prob):
            if idx in self.illegal_vars:
                continue
            var = self.tokenizer_mlm.decode(idx)
            var = var.strip()
            if var in selected_vars:
                continue
            else:
                selected_vars.append(var)
                selected_idxes.append(idx)
                selected_probs.append(prob)
                if len(selected_idxes) == k:
                    selected_probs = torch.tensor(selected_probs, dtype=torch.float)
                    selected_probs = F.softmax(selected_probs, -1)
                    selected_probs = selected_probs.detach().cpu().tolist()
                    return selected_probs, selected_idxes


    def show_top_k(self, topk_idx, topk_prob):
        for subtoken_index in range(len(topk_idx)):
            print(f'\t{subtoken_index} subtoken')
            for word, prob in zip(topk_idx[subtoken_index], topk_prob[subtoken_index]):
                word = self.tokenizer_mlm.decode(word)
                print(f'{word}: {prob:.2e} |')
