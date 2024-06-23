import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SubjectiveQA(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'subjective.csv'):
        dataset = DatasetDict()
        filename = osp.join(path, file_name)
        with open(filename) as f:
            reader = csv.reader(f, delimiter='\t')
            raw_data = []
            for row in reader:
                assert len(row) == 2
                question = row[0]
                raw_data.append({'question': question, 'answer': ''})
            dataset['dev'] = Dataset.from_list(raw_data)
        return dataset


@LOAD_DATASET.register_module()
class SubjectiveQAv2(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'subjectiveqa_v2.jsonl'):
        dataset = DatasetDict()
        filename = osp.join(path, file_name)
        with open(filename) as f:
            raw_data = f.readlines()
            raw_data = [json.loads(data) for data in raw_data]
        processed_data = []
        for source_data in raw_data:
            question = source_data['chosen'].replace('\n\nHuman:', '').strip()
            question = question.replace('\n\nAssistant:', '').rstrip()
            processed_data.append({'question': question, 'answer': ''})
        dataset['dev'] = Dataset.from_list(processed_data)
        return dataset


@LOAD_DATASET.register_module()
class SubjectiveQAv3(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'subjectiveqa_v3.jsonl'):
        dataset = DatasetDict()
        filename = osp.join(path, file_name)
        with open(filename) as f:
            raw_data = f.readlines()
            raw_data = [json.loads(data) for data in raw_data]
        processed_data = []
        for source_data in raw_data:
            question = source_data['question'].strip()
            processed_data.append({'question': question, 'answer': ''})
        dataset['dev'] = Dataset.from_list(processed_data)
        return dataset


@ICL_EVALUATORS.register_module()
class SubjectiveQAEvaluator(BaseEvaluator):

    def score(self, predictions, references):  # noqa
        """Fake Evaluator."""

        return {'score': 0.0}
