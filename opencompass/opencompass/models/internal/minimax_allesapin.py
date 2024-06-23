import json
import time
from typing import Dict, List, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['MiniMax'])
class MiniMax(BaseAPIModel):
    """Model wrapper around MiniMax.

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
                 url: str,
                 key: str,
                 max_seq_len: int = 2048,
                 call_interval: float = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None):
        self.model = path
        self.url = url
        self.headers = {
            'alles-apin-token': key,
            'Content-Type': 'application/json',
        }
        self.retry = retry
        self.countdown = retry  # track the number of retries in generate
        self.last_call_time = 0
        self.call_interval = call_interval
        self.meta_template = meta_template
        self._init_meta_template()

    def generate(
        self,
        inputs: List[str],
        max_out_len: int = 512,
        temperature: float = 0.9,
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
        assert len(inputs) == 1, 'batch_size must be 1 for MiniMax!'

        for input in inputs:
            if isinstance(input, str):
                messages = [{'sender_type': 'USER', 'text': input}]
            else:
                messages = []
                # TODO: Implement truncation in PromptList
                for item in input:
                    msg = {'text': item['prompt']}
                    if item['role'] == 'HUMAN':
                        msg['sender_type'] = 'USER'
                    elif item['role'] == 'BOT':
                        msg['sender_type'] = 'BOT'
                    messages.append(msg)

        role_meta = {'user_name': '我', 'bot_name': '学生'}

        data = {
            'model': self.model,
            'prompt': '你现在是个正在接受考试的学生，请按题干格式回答问题。',
            'role_meta': role_meta,
            'messages': messages,
            'tokens_to_generate': max_out_len,
            'top_p': 0.95,
            'temperature': temperature
        }

        raw_response = requests.post(self.url,
                                     headers=self.headers,
                                     data=json.dumps(data))
        response = raw_response.json()

        if raw_response.status_code == 200 and response[
                'msgCode'] == '10000' and response['data']['choices']:
            choices = response['data']['choices']
            msg = choices[0]['text'].strip()
            self.countdown = self.retry
        else:
            if self.countdown > 0:
                time.sleep(1)
                self.countdown -= 1
                return self.generate(inputs, max_out_len, temperature)
            else:
                print(response['msg'])
                exit(-1)

        return [msg]
