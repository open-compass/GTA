import json
from typing import Dict, Optional, Union, List

import requests
import atexit
import time
from func_timeout import func_set_timeout, FunctionTimedOut

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .openai_allesapin import OpenAIAllesAPIN

PromptType = Union[PromptList, str]


class OpenAIAssistantAllesAPIN(OpenAIAllesAPIN):
    """Model wrapper around OpenAI-AllesAPIN Assistant.

    Args:
        path (str): The name of OpenAI's model.
        url (str): URL to AllesAPIN.
        key (str): AllesAPIN key.
        instruct (str, optional): Instruction for assistant, 
            like system prompt. Defaults to None.
        tools (List[dict], optional): Tool to be used in assistant.
            Defaults to None.
        name (str, optional): Assistant name. Defaults to None.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        timeout (int) Timeout for getting run results. Defaults to 60.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 url: str,
                 key: str,
                 instruct: Optional[str] = None,
                 tools: Optional[List[dict]] = None,
                 name: Optional[str] = None,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 timeout: int = 60,
                 retry: int = 2):
        super(OpenAIAllesAPIN, self).__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         meta_template=meta_template,
                         retry=retry)
        self.url = url
        self.headers = {
            'alles-apin-token': key,
            'content-type': 'application/json',
            'OpenAI-Beta': 'assistants=v1',
        }
        if instruct is None:
            instruct = "You are a personal math tutor. When asked a question, write and run Python code to answer the question."
        # more tools should be supported later
        if tools is None:
            tools = [{"type": "code_interpreter"}]
        if name is None:
            name = 'Math Tutor'
        self._initialized = False
        asst_id = self.create_asst(path, instruct, name, tools)
        self.asst_id = asst_id
        atexit.register(self.delete_asst)
        self.timeout = timeout
       
    def create_asst(self, path, instruct, name, tools):
        if self._initialized:
            raise ValueError('Already bind to a assistant. Do not need to create.')
        url = self.url + "/assistants"
        payload = {
            "instructions": instruct,
            "name": name,
            "tools": tools,
            "model": path
        }
        raw_response = requests.request("POST", url, headers=self.headers, data=json.dumps(payload))
        try:
            response = raw_response.json()
            asst_id = response['body']['id']
            key_sign = response['data']['key-sign']
            self.key_sign = key_sign
            self.headers['key-sign'] = self.key_sign
            self._initialized = True
            self.logger.info(f'Create asst: {asst_id}, key-sign: {key_sign}')
            return asst_id
        except Exception as e:
            raise RuntimeError(f'Create assistant failed, {e}, {raw_response.text}.')

    def delete_asst(self):
        if self._initialized:
            url = self.url + f"/assistants/{self.asst_id}"
            raw_response = requests.request("DELETE", url, headers=self.headers)
            try:
                response = raw_response.json()
                status = response['body']['deleted']
                assert status
                self.logger.info(f'Delete asst: {self.asst_id}')
            except Exception as e:
                raise RuntimeError(f'Delete assistant failed, {e}, {raw_response.text}.')

    def create_thread(self):
        url = self.url + "/threads"
        payload = ''
        raw_response = requests.request("POST", url, headers=self.headers, data=json.dumps(payload))
        try:
            response = raw_response.json()
            thread_id = response['body']['id']
            self.logger.debug(f'Create thread: {thread_id}, key-sign: {self.key_sign}')
            return thread_id
        except Exception as e:
            raise RuntimeError(f'Create threads failed, {e}, {raw_response.text}.')

    def delete_thread(self, thread_id):
        url = self.url + f"/threads/{thread_id}"
        raw_response = requests.request("DELETE", url, headers=self.headers)
        try:
            response = raw_response.json()
            status = response['body']['deleted']
            assert status
            self.logger.debug(f'Delete thread: {self.asst_id}')
            return status
        except Exception as e:
            raise RuntimeError(f'Delete threads failed, {e}, {raw_response.text}.')
        
    def create_message(self, thread_id, prompt):
        url = self.url + f"/threads/{thread_id}/messages"
        payload = {
            "role": "user",  # only user is supported currently
            "content": prompt
        }
        raw_response = requests.request("POST", url, headers=self.headers, data=json.dumps(payload))
        try:
            response = raw_response.json()
            msg_id = response['body']['id']
            return msg_id
        except Exception as e:
            raise RuntimeError(f'Create message failed, {e}, {raw_response.text}.')

    def create_run(self, thread_id, asst_id):
        url = self.url + f"/threads/{thread_id}/runs"
        payload = {
            "assistant_id": asst_id
        }
        raw_response = requests.request("POST", url, headers=self.headers, data=json.dumps(payload))
        try:
            response = raw_response.json()
            run_id = response['body']['id']
            return run_id
        except Exception as e:
            raise RuntimeError(f'Create run failed, {e}, {raw_response.text}.')

    def get_run(self, thread_id, run_id):
        url = self.url + f"/threads/{thread_id}/runs/{run_id}"
        raw_response = requests.request("GET", url, headers=self.headers)
        try:
            response = raw_response.json()
            status = response['body']['status']
            return status
        except Exception as e:
            raise RuntimeError(f'Get run failed, {e}, {raw_response.text}.')

    def get_messages(self, thread_id):
        url = self.url + f"/threads/{thread_id}/messages"
        raw_response = requests.request("GET", url, headers=self.headers)
        try:
            response = raw_response.json()
            # TODO: process messages
            message_list = response['body']['data']
            return message_list
        except Exception as e:
            raise RuntimeError(f'Get messages failed, {e}, {raw_response.text}.')

    def get_completed_message(self, thread_id, run_id):

        @func_set_timeout(self.timeout)
        def wrapper():
            status = 'in_progress'
            while status != 'completed':
                status = self.get_run(thread_id, run_id)
                print(status)
                time.sleep(1)
            msg_list = self.get_messages(thread_id)
            completed_msg = ''
            for msg in msg_list[::-1]:
                if msg['run_id'] == run_id:
                    completed_msg += msg['content'][0]['text']['value'] + '\n'
            return completed_msg

        try:
            return wrapper()
        except FunctionTimedOut:
            self.logger.info(f'Function call timed out after {self.timeout}')
            return f'Function call timed out after {self.timeout}'

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
        thread_id = self.create_thread()
        assert isinstance(input, str)
        msg_id = self.create_message(thread_id, input)
        run_id = self.create_run(thread_id, self.asst_id)
        completed_msg = self.get_completed_message(thread_id, run_id)
        self.delete_thread(thread_id)
        print(completed_msg)
        return completed_msg
