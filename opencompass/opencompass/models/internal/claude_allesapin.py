import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Union, List

import requests

from opencompass.registry import MODELS
from opencompass.utils import PromptList

from ..base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class ClaudeAllesAPIN(BaseAPIModel):
    """Model wrapper around Claude-AllesAPIN.

    Args:
        path (str): The name of Claude's model.
        url (str): URL to AllesAPIN.
        key (str): AllesAPIN key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 url: str,
                 key: str,
                 query_per_second: int = 1,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        self.url = url
        self.headers = {
            'alles-apin-token': key,
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
                The PromptDict should be organized in OpenAGIEval's
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
    def _generate(self, input: str or PromptList, max_out_len: int) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))
        
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        data = {
            'model': self.path,
            'messages': messages,
            'max_tokens': max_out_len,
        }

        err_data = []
        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                raw_response = requests.post(self.url,
                                             headers=self.headers,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                time.sleep(5)
                continue
            except requests.ReadTimeout:
                time.sleep(5)
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  raw_response.content)
                continue
            if raw_response.status_code == 200 and response[
                    'msgCode'] == '10000':
                data = response['data']
                return data['content'][0]['text'].strip()
            self.logger.error(response['data'])
            err_data.append(response['data'])
            num_retries += 1

        raise RuntimeError(err_data)
