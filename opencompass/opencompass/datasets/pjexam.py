import json
import os
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class PJExamDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = []
        with open(os.path.join(path, name + '.jsonl'), 'r') as f:
            for line in f:
                line = json.loads(line)
                line['eval_infos'] = {
                    'major': line['major'],
                    'q_type': line['q_type'],
                    'std_ans': line['std_ans'],
                    'score': line['score']
                }
                dataset.append(line)
        return Dataset.from_list(dataset)


@ICL_EVALUATORS.register_module('PJExamEvaluator')
class PJExamEvaluator(BaseEvaluator):

    def _get_letters(self,
                     s,
                     candidates,
                     multiple_choice=False,
                     default_option='C'):
        if multiple_choice:
            pred = []
            for i in range(len(s)):
                if s[i] in candidates:
                    pred.append(s[i])
            if pred:
                return ''.join(sorted(set(pred)))
            else:
                return default_option
        else:
            match = re.search(f'([{candidates}])', s[::-1])
            if match:
                return match.group(1)
            return default_option

    def judge_single_choice(self, pred, refr, score):
        if pred == refr:
            return score
        else:
            return 0

    def judge_multiple_choice(self, pred, refr, score):
        if pred == refr:
            return score
        else:
            if any(i not in refr for i in pred):
                return 0
            else:
                return score // 2

    def judge_multiple_choice_optional_phys(self, pred, refr, score):
        assert score == 5
        right_count, wrong_count = 0, 0
        for i in pred:
            if i in refr:
                right_count += 1
            else:
                wrong_count += 1
        right_score = [0, 2, 4, 5][right_count]
        wrong_score = -3 * wrong_count
        return max(0, right_score + wrong_score)

    def score(self, predictions, references):
        stats = {}
        for pred, ref in zip(predictions, references):
            if ref['q_type'] == '单选题':
                pred = self._get_letters(pred, 'ABCD', multiple_choice=False)
                total = float(ref['score'])
                got = self.judge_single_choice(pred, ref['std_ans'], total)
            elif ref['q_type'] == '多选题':
                if ref['major'] == '物理' and total == 5:
                    pred = self._get_letters(pred,
                                             'ABCDE',
                                             multiple_choice=True)
                    total = float(ref['score'])
                    got = self.judge_multiple_choice_optional_phys(
                        pred, ref['std_ans'], total)
                else:
                    pred = self._get_letters(pred,
                                             'ABCD',
                                             multiple_choice=True)
                    total = float(ref['score'])
                    got = self.judge_multiple_choice(pred, ref['std_ans'],
                                                     total)
            else:
                assert 0, ref['q_type']
            stats.setdefault(ref['major'], {'total': 0, 'got': 0})
            stats[ref['major']]['total'] += total
            stats[ref['major']]['got'] += got
        for k, v in stats.items():
            stats[k]['acc'] = v['got'] / v['total'] * 100

        item = {'total': 0, 'got': 0}
        for v in stats.values():
            item['total'] += v['total']
            item['got'] += v['got']
        item['acc'] = item['got'] / item['total'] * 100
        stats['all'] = item
        return {'score': stats['all']['acc'], 'details': stats}
