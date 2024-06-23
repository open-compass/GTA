import json

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class longeval_gov_report_eDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        data = {'context': [], 'answers': []}
        with open(path, 'r', encoding='utf-8') as json_file:
            test_cases = json.load(json_file)

        for item in test_cases:
            data['context'].append(item['context'])
            data['answers'].append(item['answers'])

        dataset = Dataset.from_dict({
            'context': data['context'],
            'answers': data['answers']
        })
        return dataset
