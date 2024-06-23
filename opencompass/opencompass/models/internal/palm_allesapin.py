import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['PaLM'])
class PaLM(BaseAPIModel):
    """Model wrapper around PaLM (via Alles-APIN).

    Args:
        path (str): The name of OpenAI's model.
        key (str): Authorization key.
        max_seq_len (int): Unused here.
        call_interval (float): The minimum time interval in seconds between two
            calls to the API. Defaults to 3.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(self,
                 path: str,
                 key: str,
                 model: str = 'chat-bison-001',
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
            'alles-apin-token': key,
            'content-type': 'application/json',
        }
        self.model = model

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
    ) -> List[str]:
        """Generate results given a list of inputs.

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
            messages = [{'author': '0', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['author'] = '0'
                elif item['role'] == 'BOT':
                    msg['author'] = '1'
                messages.append(msg)

        data = {
            'model': self.model,
            'prompt': {
                'messages': messages
            },
            'temperature': 0.1,
            'candidate_count': 1
        }

        for _ in range(self.retry):
            self.wait()
            raw_response = requests.post(self.path,
                                         data=json.dumps(data),
                                         headers=self.headers)
            response = raw_response.json()
            if (raw_response.status_code == 200
                    and response['msgCode'] == '10000'):
                if 'candidates' not in response['data']:
                    print(response['data']['filters'])
                    return ''
                else:
                    msg = response['data']['candidates'][0]['content'].strip()
                    return msg

        raise RuntimeError(response['msg'])
