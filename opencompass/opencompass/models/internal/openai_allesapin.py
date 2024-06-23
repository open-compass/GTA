import json
from typing import Dict, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from ..openai_api import OpenAI

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['OpenAIAllesAPIN'])
class OpenAIAllesAPIN(OpenAI):
    """Model wrapper around OpenAI-AllesAPIN.

    Args:
        path (str): The name of OpenAI's model.
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
                 rpm_verbose: bool = False,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         meta_template=meta_template,
                         retry=retry)
        self.url = url
        self.headers = {
            'alles-apin-token': key,
            'content-type': 'application/json',
        }

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
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
            # model can be reponse with user and system
            # when it comes with agent involved.
            assert msg['role'] in ['user', 'system']
        data = {
            'model': self.path,
            'messages': messages,
        }

        for _ in range(self.retry):
            self.wait()
            raw_response = requests.post(self.url,
                                         headers=self.headers,
                                         data=json.dumps(data))
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue
            if raw_response.status_code == 200 and response[
                    'msgCode'] == '10000':
                data = response['data']
                choices = data['choices']
                if choices is None:
                    self.logger.error(data)
                else:
                    return choices[0]['message']['content'].strip()
            self.logger.error(response['msg'])

        raise RuntimeError('API call failed.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        enc = self.tiktoken.encoding_for_model(self.path)
        return len(enc.encode(prompt))
