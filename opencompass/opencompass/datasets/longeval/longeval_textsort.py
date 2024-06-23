import json
import random
import re
from itertools import permutations

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class longeval_textsortDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as json_file:
            test_cases = json.load(json_file)

            data = {
                'before': [],
                'part1': [],
                'part2': [],
                'part3': [],
                'part4': [],
                'after': [],
                'answer': []
            }
            random.seed(2680)

            for item in test_cases:
                ans = random.choice(list(permutations(range(4))))
                data['answer'].append([x + 1 for x in ans])
                chapters = [''] * len(item['chapters'])
                for i, idx in enumerate(ans):
                    chapters[idx] = item['chapters'][i]
                item['chapters'] = chapters
                data['before'].append(item['front'])
                data['part1'].append(item['chapters'][0])
                data['part2'].append(item['chapters'][1])
                data['part3'].append(item['chapters'][2])
                data['part4'].append(item['chapters'][3])
                data['after'].append(item['rear'])

        dataset = Dataset.from_dict({
            'before': data['before'],
            'part1': data['part1'],
            'part2': data['part2'],
            'part3': data['part3'],
            'part4': data['part4'],
            'after': data['after'],
            'answer': data['answer']
        })
        return dataset


@ICL_EVALUATORS.register_module()
class longeval_textsortEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        cnt = 0
        for id in range(len(predictions)):
            prediction = predictions[id]
            reference = references[id]
            order_list = re.findall(r'\d+', prediction)
            order = [
                int(re.sub('\D', '', item))  # noqa
                for item in order_list
            ]
            identical = True
            if len(reference) != len(order):
                identical = False
            for a, b in zip(reference, order):
                if a != b:
                    identical = False

            if identical:
                cnt += 1

        score = cnt / len(predictions) * 100

        return {'score': score}
