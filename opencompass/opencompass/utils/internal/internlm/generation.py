from typing import List
import inspect
import time

import torch
import torch.distributed as dist
from inference.seq_generator_module import _no_beam_search_generate
from llama.model import Transformer
from llama.tokenizer import Tokenizer


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
                     'eos_token_id': None,
                     'temperature': 1.0,
                     'top_p': 1.0,
                     'top_k': 50,
                     'do_sample': False,
                     'repetition_penalty': 1.0,
                     'min_new_tokens': 1,
                 }):
        tokenized_data = self.tokenizer(inputs,
                                        padding=True,
                                        right_align=True,
                                        return_tensors='pt')
        tokenized_data_len = tokenized_data['tokens'].shape[1]
        eos_token_id = generation_kwargs.get('eos_token_id')
        if not eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id

        sig = inspect.signature(_no_beam_search_generate)
        self.forward_kwargs['min_new_tokens'] = generation_kwargs.pop('min_new_tokens')
        forward_kwargs = {
            k: v for k, v in self.forward_kwargs.items() if k in sig.parameters
        }
        # seed
        seed = torch.tensor(time.time(), dtype=torch.int64).cuda()
        dist.broadcast(seed, src=0)
        torch.cuda.manual_seed(seed.item())
        dist.barrier()
        results = _no_beam_search_generate(
            self.model,
            tokenized_data['tokens'][..., ],
            do_sample=generation_kwargs['do_sample'],
            max_length=generation_kwargs['max_gen_len'] + tokenized_data_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            top_p=generation_kwargs['top_p'],
            top_k=generation_kwargs['top_k'],
            temperature=generation_kwargs['temperature'],
            repetition_penalty=generation_kwargs['repetition_penalty'],
            **forward_kwargs)
        results = results.squeeze(1).tolist()
        results_text = []
        for i in range(len(inputs)):
            single_res = results[i][tokenized_data_len:]
            if eos_token_id is not None:
                try:
                    single_res = single_res[:single_res.index(eos_token_id)]
                except ValueError:
                    pass
            results_text.append(self.tokenizer.tokenizer.decode(single_res))

        return results_text

    def get_logits(self, inputs):
        inputs = self.tokenizer(inputs,
                                padding=True,
                                return_tensors='pt',
                                truncation=True)
        if self.use_mask:
            outputs = self.model(input_ids=inputs['tokens'],
                                 **self.forward_kwargs)
        else:
            outputs = self.model(input_ids=inputs['tokens'])
        return outputs, inputs


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
