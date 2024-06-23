import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel


class PJLM(BaseAPIModel):
    """Model wrapper for PJLM api.

    Args:
        path (str): The name of SenseTime model.
            e.g. `ChatPJLM-latest` for the latest version of ChatPJLM.
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 999.
    """

    def __init__(
        self,
        key: str,
        url: str,
        path: str = 'ChatPJLM-latest',
        query_per_second: int = 1,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 999,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        self.token = key
        self.url = url
        self.retry = retry
        self.last_token_time = None
        self.model_name = path
        self.token_expiry_interval = 30  # 30 minutes to refresh the token

    def token_needs_renewal(self):
        # Check if the token needs renewal
        return self.token is None or self.last_token_time is None or (
            time.time() - self.last_token_time) > self.token_expiry_interval

    def renew_token(self):
        # Run the 'openxlab token' command
        result = subprocess.run(['openxlab', 'token'],
                                stdout=subprocess.PIPE,
                                text=True)

        if result.returncode == 0:
            self.token = result.stdout.strip()
            print('Token refreshed')
        else:
            print(f'Failed to refresh token. Return code: {result.returncode}')

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
        if self.token_needs_renewal():
            self.renew_token()
            self.last_token_time = time.time()

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
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
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'

                messages.append(msg)

        header = {
            'Content-Type': 'application/json',
            'Authorization': self.token
        }

        data = {
            'model': self.model_name,
            'messages': [{
                'role': 'user',
                'text': input
            }]
        }

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:
                res = requests.post(self.url,
                                    headers=header,
                                    data=json.dumps(data))
                res.raise_for_status(
                )  # Raise an exception for 4xx and 5xx status codes
                print(f'Response: {res.json()}')
                generation = res.json()['data']['choices'][0]['text']
                assert generation != ''  # Make sure the response is not empty
                return generation
            except Exception as e:
                print(f'Error in request: {e}')
                time.sleep(10)  # Wait before retrying
                max_num_retries += 1
                # continue

        print(f'Failed to generate response after {self.retry} retries.')

        raise RuntimeError(res.json().text)
