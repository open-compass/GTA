import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class Qwen(BaseAPIModel):
    """Model wrapper around Qwen.

    Documentation:
        https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-thousand-questions/

    Args:
        path (str): The name of qwen model.
            e.g. `qwen-max`
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(self,
                 path: str,
                 key: str,
                 query_per_second: int = 1,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 5,
                 max_input_chars: int = 6000,
                 generation_kwargs: Dict = {}):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        import dashscope
        dashscope.api_key = key
        self.dashscope = dashscope
        # DashScope native Generation API enforces a strict input length limit
        # (commonly: 1..6000 characters). We defensively truncate chat history
        # to avoid hard failures during long tool-augmented conversations.
        self.max_input_chars = max_input_chars

    @staticmethod
    def _messages_char_len(messages: List[Dict]) -> int:
        return sum(len((m.get('content') or '')) for m in messages)

    @classmethod
    def _truncate_messages(cls, messages: List[Dict], max_chars: int) -> List[Dict]:
        """Keep the most recent messages under a character budget.

        Preserves an initial system message (if any) and the newest turns.
        If even the newest message is too long, truncates it from the left.
        """
        if max_chars < 1:
            max_chars = 1
        if cls._messages_char_len(messages) <= max_chars:
            return messages

        system = []
        rest = messages
        if messages and messages[0].get('role') == 'system':
            system = [messages[0]]
            rest = messages[1:]

        kept_reversed: List[Dict] = []
        total = cls._messages_char_len(system)
        # Keep most recent messages first.
        for msg in reversed(rest):
            content = msg.get('content') or ''
            msg_len = len(content)
            if total + msg_len <= max_chars:
                kept_reversed.append(msg)
                total += msg_len
                continue
            # Always keep at least the newest message (truncate if needed).
            if not kept_reversed:
                available = max_chars - total
                if available < 1:
                    available = 1
                kept_reversed.append({
                    'role': msg.get('role', 'user'),
                    'content': content[-available:],
                })
                total = max_chars
            break

        kept = list(reversed(kept_reversed))
        return system + kept

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
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
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))
        """
        {
          "messages": [
            {"role":"user","content":"请介绍一下你自己"},
            {"role":"assistant","content":"我是通义千问"},
            {"role":"user","content": "我在上海，周末可以去哪里玩？"},
            {"role":"assistant","content": "上海是一个充满活力和文化氛围的城市"},
            {"role":"user","content": "周末这里的天气怎么样？"}
          ]
        }

        """

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            msg_buffer, last_role = [], None
            for index, item in enumerate(input):
                if index == 0 and item['role'] == 'SYSTEM':
                    role = 'system'
                elif item['role'] == 'BOT':
                    role = 'assistant'
                else:
                    role = 'user'
                if role != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = role
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })
        # Leave a small buffer for any server-side overhead.
        messages = self._truncate_messages(messages, max_chars=max(1, self.max_input_chars - 200))
        data = {'messages': messages}
        data.update(self.generation_kwargs)

        max_num_retries = 0
        # If the server still complains about input length, progressively shrink.
        shrink_budget = max(1, self.max_input_chars - 200)
        while max_num_retries < self.retry:
            self.acquire()
            try:
                response = self.dashscope.Generation.call(
                    model=self.path,
                    **data,
                )
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(1)
                continue

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue

            if response.status_code == 200:
                try:
                    msg = response.output.text
                    self.logger.debug(msg)
                    return msg
                except KeyError:
                    print(response)
                    self.logger.error(str(response.status_code))
                    time.sleep(1)
                    continue
            if response.status_code == 429:
                print(response)
                time.sleep(2)
                continue
            if response.status_code == 400:
                # DashScope may return 400 for multiple reasons; do not mask errors.
                resp_msg = getattr(response, 'message', '') or ''
                if 'Range of input length should be' in resp_msg:
                    # Reduce prompt budget and retry.
                    shrink_budget = max(1, shrink_budget - 500)
                    data['messages'] = self._truncate_messages(
                        data['messages'], max_chars=shrink_budget)
                    self.logger.warning(
                        'DashScope rejected input length; retry with %d chars (request %d/%d).',
                        shrink_budget, max_num_retries + 1, self.retry)
                    time.sleep(0.5)
                    max_num_retries += 1
                    continue
                if 'Input data may contain inappropriate content.' in resp_msg:
                    self.logger.warning('DashScope rejected input as inappropriate.')
                    return ''
                print('=' * 128)
                print(response)
                max_num_retries += 1
                time.sleep(1)
                continue

            if 'Range of input length should be ' in getattr(response, 'message', ''):
                print(response.message)
                return ''
            print(response)
            max_num_retries += 1

        raise RuntimeError(response.message)
