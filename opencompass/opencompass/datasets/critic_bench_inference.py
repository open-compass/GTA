import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@ICL_EVALUATORS.register_module()
class CriticBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):  # noqa
        """Fake Evaluator."""

        return {'score': 0.0}


@LOAD_DATASET.register_module()
class CriticBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             name: str = 'translate_feedback_correction.json',
             flag_name: str = 'sub',
             domain_name: str = 'translate',
             mode_name: str = 'feedback',
             reverse: bool = False,
             mappings: dict = {}):
        dataset = DatasetDict()
        file_name = osp.join(path, name)
        with open(file_name, encoding='utf-8') as f:
            data = json.load(f)
        processed_data = []
        for item in data:
            if flag_name in item:
                for key in item[flag_name]:
                    if key in ['exec_rest_a', 'exec_rest_b'] and isinstance(
                            item[flag_name][key], dict):
                        value = item[flag_name][key]['detail']
                    else:
                        value = item[flag_name][key]
                    item[key] = value
            if mode_name in ['comp_feedback'] and reverse:
                item['generation_a'], item['generation_b'] = item[
                    'generation_b'], item['generation_a']
                if domain_name == 'code_exec':
                    item['exec_rest_a'], item['exec_rest_b'] = item[
                        'exec_rest_b'], item['exec_rest_a']
            content = mappings[mode_name][domain_name].format(**item)
            processed_data.append({'question': content, 'answer': ''})
        dataset['dev'] = Dataset.from_list(processed_data)
        return dataset
