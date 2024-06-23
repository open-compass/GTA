import re
from typing import List, Union

import torch
from transformers import PreTrainedTokenizerBase

from opencompass.utils import get_logger


class LLMTokenizer(object):

    def __init__(self, tokenizer, max_seq_len=2048, tokenizer_type='llama', mode='none'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer_type = tokenizer_type

        self._set_special_tokens()

        assert mode in ['none', 'mid']
        self.mode = mode

    def __call__(self,
                 prompts,
                 padding=True,
                 right_align=False,
                 return_tensors='pt',
                 truncation=True):
        tokens = [self.encode(x, bos=True, eos=False) for x in prompts]
        if truncation:
            if self.mode == 'none':
                tokens = [i[:self.max_seq_len] for i in tokens]
            else:
                half = self.max_seq_len // 2
                new_tokens = []
                for i in tokens:
                    if len(i) <= self.max_seq_len:
                        new_tokens.append(i)
                    else:
                        new_tokens.append(i[:half] + i[-half:])
                tokens = new_tokens

        if padding:
            max_len = max([len(i) for i in tokens])
            if right_align:
                tokens = [[self.pad_token_id] * (max_len - len(i)) + i
                          for i in tokens]
            else:
                tokens = [
                    i + [self.pad_token_id] * (max_len - len(i))
                    for i in tokens
                ]

        tokens = torch.LongTensor(tokens)
        return {
            'tokens': tokens.cuda() if torch.cuda.is_available() else tokens
        }

    def encode(self, s: str, bos: bool, eos: bool):
        assert isinstance(s, str)
        s = self._process_meta_tokens(s)
        t = self._tokenize_list_str(s)
        if bos:
            t = [self.bos_token_id] + t
        if eos:
            t = t + [self.eos_token_id]
        return t

    def _process_meta_tokens(self, input_string: str) -> List[Union[str, int]]:
        # Create a pattern to match the META_TOKEN_{NUM} substrings
        pattern = re.compile(r'<META_TOKEN_(\d+)>')

        # Split the input string using the META_TOKEN_{NUM} substrings
        parts = pattern.split(input_string)

        # Combine the parts and tokens in the correct order
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text parts
                if part != '':
                    result.append(part)
            else:  # Meta token parts
                result.append(int(part))

        return result

    def _tokenize_list_str(self, s: Union[str, list]) -> List[int]:
        if isinstance(s, str):
            s = [s]
        assert isinstance(s, list)
        t = []
        for item in s:
            if isinstance(item, str):
                t += self.tokenizer.encode(item)
            elif isinstance(item, int):
                t.append(item)
            else:
                raise ValueError(f'Unsupported type {type(item)}')
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    def _set_special_tokens(self):
        if self.tokenizer_type is None or self.tokenizer_type == "":
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                self.bos_token_id = self.tokenizer.bos_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                self.bos_token_id = self.tokenizer.bos_id()
                self.eos_token_id = self.tokenizer.eos_id()
                self.pad_token_id = self.tokenizer.pad_id()
            if self.pad_token_id == -1:
                self.pad_token_id = self.bos_token_id
        elif self.tokenizer_type == 'v4':
            self.bos_token_id = self.pad_token_id = 0
            self.eos_token_id = 1
        elif self.tokenizer_type in ['llama', 'v7', 'baichuan2']:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 2
        elif self.tokenizer_type == "deepseek":
            self.bos_token_id = 32013
            self.eos_token_id = 32014
            self.pad_token_id = 32018
        else:
            raise NotImplementedError(f"Unknown tokenizer type {self.tokenizer_type}")

        # This is a hack to fit in with LLama type model
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id
        self.pad_id = self.pad_token_id

        get_logger().info(
            "Set tokenizer's bos_id: {} eos_id: {} pad_id: {}"
            .format(self.bos_id, self.eos_id, self.pad_id)
        )
