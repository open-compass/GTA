import json
import re

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class longeval_stackselectDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as json_file:
            test_cases = json.load(json_file)

            data = {'question': [], 'all_answers': [], 'answer': []}

            for item in test_cases:

                data['question'].append(item['question'])
                all_answers = ''
                for j in range(1, len(item['all_answers']) + 1):
                    all_answers += 'A' + str(
                        j) + ':\n\n' + item['all_answers'][j - 1] + '\n\n'

                data['all_answers'].append(all_answers)
                data['answer'].append(item['answer'])

        dataset = Dataset.from_dict({
            'question': data['question'],
            'all_answers': data['all_answers'],
            'answer': data['answer']
        })
        return dataset


@ICL_EVALUATORS.register_module()
class longeval_stackselectEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        cnt = 0
        for id in range(len(predictions)):
            prediction = predictions[id]
            reference = references[id]
            if len(re.findall(r'\d+', prediction)) >= 1:
                if re.findall(r'\d+',
                              prediction)[0] == re.findall(r'\d+',
                                                           reference)[0]:
                    cnt += 1

        score = cnt / len(predictions) * 100

        return {'score': score}
