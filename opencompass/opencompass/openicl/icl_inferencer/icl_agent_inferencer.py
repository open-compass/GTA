"""Agent Inferencer."""
import json
import os.path as osp
import types
from typing import List
from itertools import takewhile

from opencompass.models.lagent import LagentAgent
from opencompass.registry import ICL_INFERENCERS

from ..utils.logging import get_logger
from .icl_base_inferencer import dump_results_dict
from .icl_chat_inferencer import ChatInferencer

logger = get_logger(__name__)


class AgentInferencerOutputHandler:

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, osp.join(save_dir, filename))

    def save_results(self,
                     origin_prompt: list,
                     prediction: str,
                     idx: int,
                     gold: str = None):
        result_dict = {}
        if gold:
            result_dict['gold'] = gold
        result_dict.update({
            'prediction': prediction,
            'origin_prompt': origin_prompt,
        })
        self.results_dict[str(idx)] = result_dict

    def save_multiround_results(self,
                                origin_prompt: list,
                                prediction: str,
                                idx: int,
                                gold: str = None):
        result_dict = self.results_dict.get(str(idx), {
            'gold': [],
            'prediction': [],
            'origin_prompt': [],
            'steps': [],
        })
        result_dict['gold'].append(gold)
        result_dict['prediction'].append(prediction)
        result_dict['origin_prompt'].append(origin_prompt)
        self.results_dict[str(idx)] = result_dict


@ICL_INFERENCERS.register_module()
class AgentInferencer(ChatInferencer):
    HandlerType = AgentInferencerOutputHandler

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def get_chat_list(self,
                      ice_idx_list: List[List[int]],
                      retriever,
                      prompt_template=None) -> List[dict]:
        prompt_list = []
        input_columns = retriever.dataset_reader.input_columns

        for idx, ice_idx in enumerate(ice_idx_list):
            entry = {
                k: json.loads(item)
                for k, item in retriever.test_ds[idx].items()
                if k in input_columns
            }
            prompt_list.append(entry)
        return prompt_list


    def infer_last(self, chat: dict, index: int, output_handler):
        raise NotImplementedError

    def infer_every(self, chat: dict, index: int, output_handler):
        dialogs = chat['dialogs']
        user_indices = [
            i for i, item in enumerate(dialogs) if item['role'] == 'user'
        ]

        memory = None
        for i in user_indices:
            steps, memory = self.model.chat(
                query=dialogs[i]['content'],
                memory=memory,
                resources=chat.get('resources'),
            )
            output_handler.save_multiround_results(
                origin_prompt=dialogs[i]['content'],
                prediction=steps,
                gold=list(
                    takewhile(lambda i: i['role'] != 'user', dialogs[i + 1:])),
                idx=index,
            )

        self.model.reset()

    def infer_every_with_gt(self, chat: dict, index: int, output_handler):
        dialogs = chat['dialogs']
        assistant_indices = [
            i for i, item in enumerate(dialogs) if item['role'] == 'assistant'
        ]

        for idx in range(len(assistant_indices)):
            i = assistant_indices[idx]
            stop = True if idx == len(assistant_indices) - 1 else False
            step = self.model.next_step(
                history=dialogs[:i],
                resources=chat.get('resources'),
                stop=stop
            )
            output_handler.save_multiround_results(
                origin_prompt=dialogs[:i],
                prediction=step,
                gold=dialogs[i],
                idx=index,
            )
            self.model.reset()
