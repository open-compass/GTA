import json

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class longeval_trec_eDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as json_file:
            test_cases = json.load(json_file)
            data = {'input': [], 'context': [], 'all_labels': []}

            for item in test_cases:
                data['input'].append(item['input'])
                data['context'].append(item['context'])
                data['all_labels'].append({
                    'answers': item['answers'],
                    'all_classes': item['all_classes']
                })

        dataset = Dataset.from_dict({
            'input': data['input'],
            'context': data['context'],
            'all_labels': data['all_labels']
        })
        return dataset
