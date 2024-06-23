# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
from typing import List

import torch
from llama.model import Transformer
from llama.tokenizer import Tokenizer

from .generation import _no_beam_search_generate

# TODO: find out why reserved 40G, allocated 80G
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class LLAMAGenerator:

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        generation_kwargs={'max_gen_len': 100},
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> List[str]:
        #import pdb; pdb.set_trace()
        max_gen_len = generation_kwargs['max_gen_len']
        bsz = len(prompts)
        prompts_len = [len(i) for i in prompts]
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in prompts
        ]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len),
                            self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            # truncuate from left hand side
            tokens[k, :min(total_len, len(t))] = torch.tensor(
                t[-min(total_len, len(t)):]).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos],
                                        prev_pos)[:, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos],
                                     tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t)[prompts_len[i]:])
        return decoded

    def get_logits(self, prompts):
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # 0 for ppl evaluation
        self.tokenizer.pad_id = self.tokenizer.pad_token_id = 0
        prompt_tokens = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in prompts
        ]
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(params.max_seq_len, max_prompt_size)

        tokens = torch.full((bsz, total_len),
                            self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            # truncuate from left hand side
            tokens[k, :min(total_len, len(t))] = torch.tensor(
                t[-min(total_len, len(t)):]).long()

        outputs = self.model.forward(tokens, 0)
        return outputs, {'tokens': tokens}


class LLMGenerator:

    def __init__(self, model, tokenizer, use_mask=False, forward_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.use_mask = use_mask
        self.forward_kwargs = forward_kwargs

    def generate(self,
                 inputs,
                 generation_kwargs={
                     'max_gen_len': 100,
                     'eos_token_id': None
                 }):
        prompt_len = [len(i) for i in inputs]
        tokenized_data = self.tokenizer(
            inputs, padding=True, right_align=True, return_tensors='pt')
        tokenized_data_len = tokenized_data['tokens'].shape[1]
        padding_data = self.tokenizer.tokenizer.decode(
            tokenized_data['tokens'].tolist())
        eos_token_id = generation_kwargs.get('eos_token_id')
        if not eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id
        results = _no_beam_search_generate(
            self.model,
            tokenized_data['tokens'][..., ],
            do_sample=False,
            max_length=generation_kwargs['max_gen_len'] + tokenized_data_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            **self.forward_kwargs)
        results = results.squeeze(1).tolist()
        results_text = [
            self.tokenizer.tokenizer.decode(results[i])[len(padding_data[i]):]
            for i in range(len(inputs))
        ]

        def trunc_eos(text):
            eos_text = self.tokenizer.tokenizer.decode([eos_token_id])
            try:
                text = text[:text.index(eos_text)]
            except ValueError:
                pass
            return text

        # TODO: fix this hack
        if generation_kwargs.get('eos_token_id') is not None:
            results_text = [trunc_eos(t) for t in results_text]
        return results_text

    def get_logits(self, inputs):
        inputs = self.tokenizer(
            inputs, padding=True, return_tensors='pt', truncation=True)
        if self.use_mask:
            outputs = self.model(
                input_ids=inputs['tokens'], **self.forward_kwargs)
        else:
            outputs = self.model(input_ids=inputs['tokens'])
        return outputs, inputs


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
