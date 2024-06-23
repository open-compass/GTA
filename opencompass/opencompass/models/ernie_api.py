import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['ERNIE'])
class ERNIE(BaseAPIModel):
    """Model wrapper around ERNIE.

    Args:
        path (str): Path to ERNIE.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        self.headers = {
            'content-type': 'application/json',
        }

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        return results

    def _generate(
        self,
        input: str or PromptList,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            input = input[-self.max_seq_len:]
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            # TODO: Implement truncation in PromptList
            word_ctr = 0
            for item in input:
                msg = {'content': item['prompt']}
                if word_ctr >= self.max_seq_len:
                    break
                if len(msg['content']) + word_ctr > self.max_seq_len:
                    msg['content'] = msg['content'][word_ctr -
                                                    self.max_seq_len:]
                word_ctr += len(msg['content'])
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                messages.append(msg)
            # in case the word break results in even number of messages
            if len(messages) > 0 and len(messages) % 2 == 0:
                messages = messages[:-1]

        data = {'messages': messages, 'stream': False}

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            raw_response = requests.post(self.path,
                                         data=json.dumps(data),
                                         headers=self.headers)
            response = raw_response.json()
            if (raw_response.status_code == 200
                    and 'error_code' not in response):
                msg = response['result'].strip()
                return msg
            if response['error_code'] != 18:
                num_retries += 1

        raise RuntimeError(response['error_msg'])
