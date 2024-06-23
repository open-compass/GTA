import sys
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['LLM'])
class LLM(BaseModel):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None):
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
        from ...utils.internal.load_model import load_llm
        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type)

    def _load_tokenizer(self, tokenizer_path: str, tokenizer_type: str,
                        max_seq_len: int):
        from sentencepiece import SentencePieceProcessor

        from ...utils.internal.load_model import LLMTokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        tokenizer = LLMTokenizer(tokenizer,
                                 max_seq_len=max_seq_len,
                                 tokenizer_type=tokenizer_type)
        self.tokenizer = tokenizer

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        tokens = self.tokenizer([prompt], truncation=False)['tokens']
        return len(tokens[0])

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        return self.generator.generate(inputs,
                                       generation_kwargs={
                                           'max_gen_len': max_out_len,
                                           'eos_token_id': self.eos_token_id
                                       })

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
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss


@MODELS.register_module(name=['LLMv2'])
class LLMv2(LLM):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 model_type: Optional[str] = 'converted',
                 tokenizer_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None,
                 module_path: Optional[str] = None):
        if module_path is not None:
            sys.path.insert(0, module_path)
            sys.path.insert(0, os.path.join(module_path, 'internlm'))
        assert model_type in [
            'origin', 'converted'
        ], 'The model type provided is invalid. Please ensure that the model type is either origin or converted. '  # noqa: E501

        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             model_type=model_type,
                             tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    model_type: Optional[str] = None,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
        from opencompass.utils.internal.load_model import load_llm
        from opencompass.utils.internal.model2.converted_llama.packed_pipeline_flash_converted_llama1d import \
            Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D  # noqa: E501
        from opencompass.utils.internal.model2.llama.packed_pipeline_flash_llama1d import \
            Packed_Flash_LLAMA_exlarge_pipeline_1D
        module = None
        if model_type == 'origin':
            module = Packed_Flash_LLAMA_exlarge_pipeline_1D
        elif model_type == 'converted':
            module = Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D
        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=module)


@MODELS.register_module(name=['LLMv3'])
class LLMv3(LLM):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 model_type: Optional[str] = 'converted',
                 tokenizer_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None):
        assert model_type in [
            'origin', 'converted'
        ], 'The model type provided is invalid. Please ensure that the model type is either origin or converted. '  # noqa: E501

        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             model_type=model_type,
                             tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    model_type: Optional[str] = None,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
        from opencompass.utils.internal.load_model import load_llm
        from opencompass.utils.internal.model3.converted_llama.packed_pipeline_flash_converted_llama1d import \
            Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D  # noqa: E501
        from opencompass.utils.internal.model3.llama.packed_pipeline_flash_llama1d import \
            Packed_Flash_LLAMA_exlarge_pipeline_1D
        module = None
        if model_type == 'origin':
            module = Packed_Flash_LLAMA_exlarge_pipeline_1D
        elif model_type == 'converted':
            module = Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D
        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=module)

class LLMv4(LLM):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 model_type: Optional[str] = 'converted',
                 tokenizer_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None):
        assert model_type in [
            'origin', 'converted'
        ], 'The model type provided is invalid. Please ensure that the model type is either origin or converted. '  # noqa: E501

        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             model_type=model_type,
                             tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    model_type: Optional[str] = None,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
        from trainllm.load_model import load_llm
        from trainllm.model4.converted_llama2.packed_pipeline_flash_converted_llama1d2 import \
            Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D2  # noqa: E501
        from trainllm.model4.llama.packed_pipeline_flash_llama1d import \
            Packed_Flash_LLAMA_exlarge_pipeline_1D
        module = None
        if model_type == 'origin':
            module = Packed_Flash_LLAMA_exlarge_pipeline_1D
        elif model_type == 'converted':
            module = Packed_Flash_Converted_LLAMA_exlarge_pipeline_1D2
        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=module)
