import json
import re

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class longeval_linesDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as json_file:
            json_list = list(json_file)

            data = {'prompt': [], 'expected_number': [], 'random_idx': []}
            for test_case in json_list:
                test_case = json.loads(test_case)
                data['prompt'].append(test_case['prompt'])
                data['expected_number'].append(test_case['expected_number'])
                data['random_idx'].append(test_case['random_idx'][0])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'expected_number': data['expected_number'],
            'random_idx': data['random_idx']
        })
        return dataset


@ICL_EVALUATORS.register_module()
class longeval_linesEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        cnt = 0
        for id in range(len(predictions)):
            prediction = predictions[id]
            reference = references[id]

            # TODO: need to fix the problem caused by the introduction
            #  of the following code
            prediction = prediction.strip().split('\n')[0]
            # match = re.search('<(.*?)>', prediction, re.DOTALL)
            # if match:
            #     prediction = match.group(1)
            # else:
            #     prediction = prediction.split('>')[0]
            #     prediction = prediction.split('.')[0]

            response_number = re.findall(r'\d+', prediction)
            if response_number is not None and len(response_number) > 0:
                response_number = int(response_number[-1])
            else:
                response_number = -1

            cnt += (reference == response_number)

        score = cnt / len(predictions) * 100

        return {'score': score}
