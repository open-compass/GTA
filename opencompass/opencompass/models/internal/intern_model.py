import importlib
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch
from opencompass.utils.logging import get_logger

from opencompass.registry import MODELS
from opencompass.models.base import BaseModel, LMTemplateParser

module_info = {
    'LLAMA': ('model.modeling_llama', 'build_model_with_cfg'),
    'LLAMA_TF32': ('model.modeling_llama_tf32', 'build_model_with_cfg'),
    'INTERNLM': ('internlm.model.modeling_internlm', 'build_model_with_cfg'),
    "LLAMA2": ("internlm.model.modeling_llama", "build_model_with_cfg"),
    "BAICHUAN2": ("model.modeling_baichuan2", "build_model_with_cfg"),
    "H3": ("model.modeling_h3", "build_h3_with_cfg"),
    "Hyena": ("model.modeling_hyena", "build_hyena_with_cfg"),
    "MISTRAL": ("model.modeling_mistral", "build_model_with_cfg"),
    "LLAMA_NORMHEAD": ("model.modeling_llama_normhead", "build_model_with_cfg"),
    "INTERNLM2": ("model.modeling_internlm2", "build_model_with_cfg")
}
tokenizer_type_dict = {
    "v4": ["v4"],
    "llama": ["v7", "llama", "v11", "baichuan2", "v13"],
}


def import_module(module_name, attribute_name=None):
    try:

        module = importlib.import_module(module_name)

        if attribute_name:
            attribute = getattr(module, attribute_name)
            return attribute
        else:
            return module
    except ImportError:
        print(f'无法导入模块: {module_name}')
        traceback.print_exc()
        raise

@MODELS.register_module()
class InternLMwithModule(BaseModel):

    def __init__(self,
                 path: str,
                 module_path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 model_config: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 model_type: Optional[str] = 'LLAMA',
                 ckpt_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None,
                 model_dtype: Optional[str] = None,
                 w2w3_bug: bool = False,
                 submit_time: Optional[str] = None,
                 generation_kwargs={},
                 sync_rank: bool = False,
                 mode='none'):
        sys.path.insert(0, module_path)
        sys.path.insert(0, os.path.join(module_path, 'internlm'))

        # in some 123B model trained from some early version of
        # `train_internlm`, it has a w2 and w3 error, which may get
        # `miss match error` when `load_llm` now. Set
        # `w2w3_bug=True` can fix this error.
        self.w2w3_bug = w2w3_bug

        # TODO: mode is not a good name, change it both here and huggingface.py
        # mode = 'mid' is used only in longtext eval, which cut off tokens in
        # the middle
        # https://github.com/THUDM/LongBench
        assert mode in ['none', 'mid']
        self.mode = mode

        self.logger = get_logger()

        self.check_modified_time(module_path, model_config, submit_time)
        _default_generation_kwargs = {
            'temperature': 1.0,
            'top_p': 1.0,
            'top_k': 50,
            'do_sample': False,
            'repetition_penalty': 1.0,
        }
        self.generation_kwargs = _default_generation_kwargs.copy()
        self.generation_kwargs.update(generation_kwargs)
        self.logger.info(f'generation_kwargs: {self.generation_kwargs}')

        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(
                path=path,
                max_seq_len=max_seq_len,
                tokenizer_path=tokenizer_path,
                tokenizer_type=tokenizer_type,
                model_config=model_config,
                model_type=model_type,
                model_dtype=model_dtype,
                w2w3_bug=self.w2w3_bug,
                ckpt_type=ckpt_type
            )
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        self.sync_rank = sync_rank

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None,
                    model_config: Optional[str] = None,
                    model_type: Optional[str] = None,
                    model_dtype: Optional[str] = None,
                    ckpt_type: Optional[str] = None,
                    w2w3_bug: bool = False):
        from opencompass.utils.internal.internlm import load_llm
        module_name, attribute_name = module_info[model_type]
        module = import_module(module_name, attribute_name)
        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=module,
            model_type=model_type,
            ckpt_type=ckpt_type,
            model_config_path=model_config,
            model_dtype=model_dtype,
            w2w3_bug=w2w3_bug,
            mode=self.mode,
        )

    def _load_tokenizer(self, tokenizer_path: str, tokenizer_type: str,
                        max_seq_len: int):
        from sentencepiece import SentencePieceProcessor

        from opencompass.utils.internal.internlm import LLMTokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        tokenizer = LLMTokenizer(tokenizer,
                                 max_seq_len=max_seq_len,
                                 tokenizer_type=tokenizer_type,
                                 mode=self.mode)
        self.tokenizer = tokenizer

    def check_modified_time(self, module_path, model_config, submit_time):
        if submit_time is None:
            return
        time_format = "%Y/%m/%d %H:%M"
        modified_files = set()

        def _check_file_modified(filepath, submit_time):
            submit_timestamp = time.mktime(time.strptime(submit_time, time_format))
            _, suffix = os.path.splitext(filepath)
            if suffix != ".py":
                return
            if os.path.getmtime(filepath) > submit_timestamp:
                modified_files.add(filepath)

        def _check_dir_modified(dirpath, submit_time):
            for root, dirs, files in os.walk(dirpath):
                for filename in files:
                    _check_file_modified(os.path.join(root, filename), submit_time)

        # check model_config train_internlm/intenrlm train_intenrlm/model
        _check_file_modified(model_config, submit_time)
        _check_dir_modified(os.path.join(module_path, "internlm"), submit_time)
        _check_dir_modified(os.path.join(module_path, "model"), submit_time)

        if len(modified_files) != 0:
            self.logger.error(
                "The following files are modified after being submitted "
                f"at: {modified_files}"
            )
            exit(1)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        tokens = self.tokenizer([prompt], truncation=False)['tokens']
        return len(tokens[0])

    def generate(self, inputs: List[str], max_out_len: int, min_out_len: Optional[int] = None) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = self.generation_kwargs.copy()
        if min_out_len is None:
            # keep same with train_internlm's default value
            min_out_len = 1
        generation_kwargs.update({'max_gen_len': max_out_len,
                                  'eos_token_id': self.eos_token_id,
                                  'min_new_tokens': min_out_len})
        return self.generator.generate(inputs,
                                       generation_kwargs=generation_kwargs)

    def get_ppl(self,
                input_texts: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            input_texts (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out.

        Returns:
            List[float]: A list of perplexity scores.
        """
        outputs, inputs = self.generator.get_logits(input_texts)

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs['tokens'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(
            self, input_texts: List[str], conts: List[str]) -> List[float]:
        outputs, inputs = self.generator.get_logits(input_texts)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs['tokens'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        replaced_texts = [input_text.replace(cont, '') for input_text, cont in zip(input_texts, conts)]
        replaced_lens = [len(self.encode(input_text)[0]) for input_text in replaced_texts]
        loglikelihoods = []
        for nloss, nlen, rlen in zip(loss, lens, replaced_lens):
            nlen, rlen = int(nlen), int(rlen)
            nloss = nloss[:nlen]
            nloss = nloss[rlen:].float().sum().cpu().detach().numpy()
            loglikelihoods.append(-nloss)
        return np.array(loglikelihoods)

    def get_mink_percent(self, input_texts: List[str], k: int=20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""
        outputs, inputs = self.generator.get_logits(input_texts)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs['tokens'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = - nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def encode(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer([prompt], truncation=False, return_tensors='pt')['tokens']
        return tokens

    def decode(self, tokens: torch.Tensor) -> str:
        inputs = self.tokenizer.tokenizer.decode(tokens.tolist())
        return inputs[0]
