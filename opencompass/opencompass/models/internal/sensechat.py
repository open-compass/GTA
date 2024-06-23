import json
import time
from typing import Dict, List, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['SenseChat'])
class SenseChat(BaseAPIModel):
    """Model wrapper around SenseChat models.

    Args:
        path (str): The name of OpenAI's model.
        key (str): Authorization key for SenseChat API.
        user (str): User name for SenseChat API.
        max_seq_len (int): Unused here.
        call_interval (float): The minimum time interval in seconds between two
            calls to the API. Defaults to 3.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 key: str,
                 user: str,
                 max_seq_len: int = 2048,
                 call_interval: float = 3,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None):
        self.model = path
        self.headers = {
            'Authorization': f'{key}',
            'Content-Type': 'text/event-stream',
            'Connection': 'keep-alive',
        }
        self.user = user
        self.retry = retry
        self.countdown = retry  # track the number of retries in generate
        self.last_call_time = 0
        self.call_interval = call_interval
        self.meta_template = meta_template
        self._init_meta_template()

    def generate(
        self,
        inputs: List[str],
        max_out_len: int = 2048,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.8,
    ) -> List[str]:
        """Generate summaries from a list of inputs.

        Args:
            inputs (List[str]): A batch of strings.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature o use,
                Defaults to 0.9.

        Returns:
            List[str]: A list of generated strings.
        """
        # hold for 3 secs
        current_time = time.time()
        time_elapsed = current_time - self.last_call_time

        if time_elapsed < self.call_interval:
            time_to_wait = self.call_interval - time_elapsed
            time.sleep(time_to_wait)

        self.last_call_time = time.time()

        assert isinstance(inputs, list)
        assert len(inputs) == 1, 'batch_size must be 1 for SenseChat!'
        if isinstance(inputs[0], str):
            inputs = [{'role': 'user', 'content': inputs[0]}]

        data = {
            'messages': inputs,
            'temperature': temperature,
            'top_p': 0.7,
            'max_new_tokens': max_out_len,
            'repetition_penalty': 1,
            'stream': False,
            'user': f'{self.user}'
        }

        response = requests.post(self.model,
                                 headers=self.headers,
                                 data=json.dumps(data))

        if response.status_code == 200:
            msg = response.json()['data']['choices'][0]['message'].strip()
            self.countdown = self.retry
        else:
            if self.countdown > 0:
                self.countdown -= 1
                return self.generate(inputs, max_out_len, temperature)
            else:
                msg = 'ERROR: API call failed.'
                self.countdown = self.retry

        return [msg]

    def get_token_len(self, strs: Union[List[str],
                                        str]) -> Union[List[int], int]:
        is_batched = isinstance(strs, list)
        if not is_batched:
            strs = [strs]
        return len(strs[0]) if not is_batched else [len(s) for s in strs]

    def to(self, device):
        pass
