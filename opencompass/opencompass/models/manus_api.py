import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class ManusAPI(BaseAPIModel):
    """Model wrapper around Manus Tasks API.

    This wrapper adapts Manus' async task API (POST /v1/tasks) to OpenCompass'
    synchronous `generate()` interface by polling task status until completion.

    Expected (from user-provided example):

    - Create task:
        POST https://api.manus.ai/v1/tasks
        headers: {"API_KEY": "...", "accept": "application/json", "content-type": "application/json"}
        body: {"prompt": "..."}

    Because upstream response schema may vary, this implementation is defensive
    and tries several common fields for `task_id`, status, and output.

    Args:
        key (str): Manus API key. If set to "ENV", it is read from env var
            MANUS_API_KEY.
        api_base (str): API base url, defaults to "https://api.manus.ai/v1".
        create_endpoint (str): Create-task endpoint path, defaults to "/tasks".
        get_endpoint_tmpl (str): Poll endpoint path template, defaults to
            "/tasks/{task_id}".
        poll_interval (float): Seconds between polls.
        timeout (float): Max seconds to wait for one task.
        extra_headers (dict): Additional headers merged into request headers.
        extra_body (dict): Additional fields merged into create-task request body.
    """

    def __init__(
        self,
        path: str = 'manus',
        key: str = 'ENV',
        api_base: str = 'https://api.manus.ai/v1',
        create_endpoint: str = '/tasks',
        get_endpoint_tmpl: str = '/tasks/{task_id}',
        query_per_second: int = 1,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        poll_interval: float = 0.5,
        timeout: float = 120.0,
        extra_headers: Optional[Dict] = None,
        extra_body: Optional[Dict] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )

        if key == 'ENV':
            key = os.getenv('MANUS_API_KEY', '')
        if not key:
            raise ValueError(
                'ManusAPI requires an API key. Pass `key=...` or set env var MANUS_API_KEY.'
            )

        self.api_base = api_base.rstrip('/')
        self.create_url = f'{self.api_base}{create_endpoint}'
        self.get_endpoint_tmpl = get_endpoint_tmpl
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.extra_body = extra_body or {}

        base_headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'API_KEY': str(key),
        }
        if extra_headers:
            base_headers.update(extra_headers)
        self.headers = base_headers

    def generate(self, inputs: List[PromptType], max_out_len: int = 512) -> List[str]:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._generate, inputs, [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(self, input: PromptType, max_out_len: int = 512) -> str:
        assert isinstance(input, (str, PromptList))

        prompt = self._to_prompt_string(input)

        payload = {
            'prompt': prompt,
            **self.extra_body,
        }

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                create_resp = requests.post(self.create_url, json=payload, headers=self.headers, timeout=60)
            except Exception as err:
                self.release()
                self.logger.error(f'Manus create-task request error: {err!r}')
                time.sleep(2)
                max_num_retries += 1
                continue

            try:
                create_json = create_resp.json()
            except Exception as err:
                self.release()
                self.logger.error(f'Manus create-task response parse error: {err!r}')
                time.sleep(1)
                max_num_retries += 1
                continue
            self.release()

            if create_resp.status_code >= 400:
                self.logger.error(f'Manus create-task failed: {create_resp.status_code} {create_json}')
                time.sleep(1)
                max_num_retries += 1
                continue

            # If API returns immediate output, use it.
            immediate = self._extract_output(create_json)
            if immediate:
                return self._truncate(immediate, max_out_len)

            task_id = self._extract_task_id(create_json)
            if not task_id:
                # No task_id and no output -> can't proceed.
                self.logger.error(f'Manus create-task response missing task id/output: {create_json}')
                max_num_retries += 1
                continue

            try:
                final_text = self._poll_task(task_id)
                return self._truncate(final_text, max_out_len)
            except Exception as err:
                self.logger.error(f'Manus poll error: {err!r}')
                max_num_retries += 1
                time.sleep(1)

        raise RuntimeError('Calling Manus API failed after retrying. Check logs for details.')

    def _poll_task(self, task_id: str) -> str:
        end_time = time.time() + float(self.timeout)
        url = f"{self.api_base}{self.get_endpoint_tmpl.format(task_id=task_id)}"

        last_json = None
        while time.time() < end_time:
            self.acquire()
            try:
                resp = requests.get(url, headers=self.headers, timeout=60)
            finally:
                self.release()

            try:
                data = resp.json()
            except Exception:
                data = None

            if resp.status_code >= 400:
                raise RuntimeError(f'Polling task failed: {resp.status_code} {data}')

            last_json = data
            if isinstance(data, dict):
                output = self._extract_output(data)
                if output:
                    return output

                status = self._extract_status(data)
                if status in {'completed', 'succeeded', 'success', 'finished', 'done'}:
                    # Completed but no recognizable output field.
                    return ''
                if status in {'failed', 'error', 'cancelled', 'canceled'}:
                    raise RuntimeError(f'Task ended with status={status}: {data}')

            time.sleep(self.poll_interval)

        raise TimeoutError(f'Timeout waiting for Manus task {task_id}. Last response: {last_json}')

    @staticmethod
    def _extract_task_id(data: Dict) -> Optional[str]:
        # Common patterns.
        for key in ('task_id', 'taskId', 'id'):
            v = data.get(key)
            if isinstance(v, str) and v:
                return v
        # Nested patterns.
        if isinstance(data.get('data'), dict):
            for key in ('task_id', 'taskId', 'id'):
                v = data['data'].get(key)
                if isinstance(v, str) and v:
                    return v
        return None

    @staticmethod
    def _extract_status(data: Dict) -> str:
        for key in ('status', 'state'):
            v = data.get(key)
            if isinstance(v, str):
                return v.lower()
        if isinstance(data.get('data'), dict):
            for key in ('status', 'state'):
                v = data['data'].get(key)
                if isinstance(v, str):
                    return v.lower()
        return ''

    @staticmethod
    def _extract_output(data: Dict) -> str:
        # Try a few common locations.
        candidates = []
        for key in ('output', 'result', 'text', 'completion', 'answer', 'content'):
            candidates.append(data.get(key))
        if isinstance(data.get('data'), dict):
            for key in ('output', 'result', 'text', 'completion', 'answer', 'content'):
                candidates.append(data['data'].get(key))
        if isinstance(data.get('message'), dict):
            candidates.append(data['message'].get('content'))
        for c in candidates:
            if isinstance(c, str) and c.strip():
                return c.strip()
        return ''

    @staticmethod
    def _to_prompt_string(input: PromptType) -> str:
        if isinstance(input, str):
            return input

        parts: List[str] = []
        for item in input:
            role = item.get('role')
            text = item.get('prompt', '')
            if not isinstance(text, str):
                text = str(text)
            if role == 'SYSTEM':
                tag = 'SYSTEM'
            elif role == 'HUMAN':
                tag = 'USER'
            elif role == 'BOT':
                tag = 'ASSISTANT'
            else:
                tag = str(role or 'USER')
            if text.strip():
                parts.append(f'{tag}: {text.strip()}')
        return '\n\n'.join(parts) if parts else ''

    @staticmethod
    def _truncate(text: str, max_out_len: int) -> str:
        # OpenCompass' API models usually truncate by tokens; here we keep it simple.
        if max_out_len <= 0:
            return ''
        # Heuristic: ~4 chars per token; keep slightly more generous for CJK.
        char_limit = max_out_len * 6
        return text if len(text) <= char_limit else text[:char_limit]
