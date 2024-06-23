from typing import Dict, Optional, Union

from opencompass.models.base import LMTemplateParser
from opencompass.models.internal.pjlm import LLM
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['LLama'])
class LLama(LLM):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 max_batch_size: int = 16,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None):
        assert not tokenizer_only, 'LLama does not support tokenizer only mode'
        self._load_model(path=path,
                         max_seq_len=max_seq_len,
                         max_batch_size=max_batch_size,
                         tokenizer_path=tokenizer_path,
                         tokenizer_type=tokenizer_type)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
        from opencompass.utils.internal.load_model import load_llama
        self.model, self.tokenizer, self.generator = load_llama(
            path,
            max_seq_len,
            max_batch_size,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type)

    def get_token_len(self,
                      prompt: str,
                      bos: bool = True,
                      eos: bool = True) -> int:
        return len(self.tokenizer.encode(prompt, bos=bos, eos=eos))
