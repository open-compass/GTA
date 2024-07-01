import json
import os
from pathlib import Path
from typing import List
import copy
import re

import numpy as np
from datasets.arrow_dataset import Dataset
from sentence_transformers import SentenceTransformer, util

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


def get_all_file_paths(directory: str) -> list:
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def organize_dialogs(sample: dict, path: str) -> List[dict]:
    dialogs = []
    file_paths = get_all_file_paths(path)
    for item in sample['dialogs']:
        if item['role'] == 'tool':
            dialog = dict(
                role='tool',
                name=item['name'],
                content=item['content'],
            )
            dialogs.append(dialog)
        elif item['role'] == 'assistant' and 'tool_calls' in item.keys():
            dialog = copy.deepcopy(item)
            for name, value in dialog['tool_calls'][0]['function']['arguments'].items():
                if isinstance(value, str) and os.path.join(path, value) in file_paths:
                    dialog['tool_calls'][0]['function']['arguments'][name] = os.path.join(path, value)
            dialogs.append(dialog)
        else:
            dialogs.append(item)

    return dialogs


@LOAD_DATASET.register_module()
class GTABenchDataset(BaseDataset):
    """GTA Benchmark."""

    @staticmethod
    def load(path: str):
        data_root = Path(path)
        data_file = data_root / 'dataset.json'
        assert os.path.exists(data_file), f'Path {path} does not exist.'

        data = json.load(open(data_file))
        data_list = []
        for idx, item in data.items():
            idx = int(idx)
            tools = [
                dict(type='tool', name=tool['name']) for tool in item['tools']
            ]
            files = [
                dict(type='file',
                     filetype=file['type'],
                     path=str((data_root / file['path']).absolute()))
                for file in item['files']
            ]
            gt_answer = item['gt_answer']
            sample = {
                'dialogs': json.dumps(organize_dialogs(item, str(data_root.absolute()))),
                'resources': json.dumps(tools + files),
                'gt_answer': json.dumps(gt_answer)
            }
            data_list.append(sample)
        dataset = Dataset.from_list(data_list)

        return dataset


@ICL_EVALUATORS.register_module()
class GTABenchEvaluator(BaseEvaluator):

    def __init__(self, mode) -> None:
        assert mode in ['every', 'every_with_gt']
        self.mode = mode
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

    def bert_score(self, pred: str, gt: str) -> float:
        pred_emb = self.sentence_model.encode(pred, convert_to_tensor=True)
        gt_emb = self.sentence_model.encode(gt, convert_to_tensor=True)
        score = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
        return score[0][0]

    @staticmethod
    def get_response_type(item):
        if 'tool_calls' in item:
            return 'tool', item['tool_calls'][0]['function']
        elif item['role'] == 'assistant':
            return 'answer', item['content']
        else:
            return 'tool_return', item['content']

    @staticmethod
    def iscorrect(pred: str, ref: dict):
        count = 0
        for aliases in ref['whitelist']:
            pattern = r'\b(?:' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
            if re.search(pattern, pred, re.IGNORECASE):
                count += 1
        if not ref['blacklist']:
            if count == len(ref['whitelist']):
                return True
        else:
            pattern_bk = r'\b(?:' + '|'.join(re.escape(alias) for aliases in ref['blacklist'] for alias in aliases) + r')\b'
            if count == len(ref['whitelist']) and not re.search(pattern_bk, pred, re.IGNORECASE):
                return True
        return False

    def simscore(self, pred: str, ref: list):
        max_score = 0
        for s in ref:
            score = self.bert_score(pred, s)
            if score > max_score:
                max_score = score
        return max_score
    
    @staticmethod
    def gettype(name: str):
        perception = ['OCR', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox']
        operation = ['DrawBox', 'AddText', 'GoogleSearch']
        logic = ['Calculator', 'Solver', 'Plot', 'MathOCR', 'CountGivenObject']
        creativity = ['TextToImage', 'ImageStylization']
        if name in perception:
            return 'perception'
        elif name in operation:
            return 'operation'
        elif name in logic:
            return 'logic'
        elif name in creativity:
            return 'creativity'
        else:
            return 'none'

    def score(self, predictions: list, references: list, gold: list):
        if self.mode == 'every_with_gt':
            total = {'tool': 0, 'answer': 0}
            metrics = {
                'inst_align': 0,
                'tool_acc': 0,
                'arg_acc': 0,
                'answer_acc': 0,
                'tool_call': 0,
                'tool_call_error': 0
            }
            for preds, gts, ref in zip(predictions, gold, references):
                ref = json.loads(ref)
                if ref:
                    total['answer'] += 1
                for pred, gt in zip(preds, gts):
                    pred_type, pred_ = self.get_response_type(pred)
                    gt_type, gt_ = self.get_response_type(gt)
                    if pred_type == gt_type and 'error' not in pred:
                        metrics['inst_align'] += 1
                    if gt_type == 'tool':
                        total['tool'] += 1
                    if pred_type == 'tool':
                        metrics['tool_call'] += 1
                        if 'error' in pred:
                            metrics['tool_call_error'] += 1
                    if pred_type == gt_type == 'tool' and pred_['name'] == gt_[
                            'name']:
                        metrics['tool_acc'] += 1
                        if pred_['arguments'] == gt_['arguments']:
                            metrics['arg_acc'] += 1
                    elif pred_type == gt_type == 'answer':
                        if isinstance(ref, dict):
                            metrics['answer_acc'] += self.iscorrect(pred_, ref)
                        elif isinstance(ref, list):
                            metrics['answer_acc'] += self.simscore(pred_, ref)
                            
            return dict(
                inst_align=metrics['inst_align'] / sum(total.values()) * 100,
                tool_acc=metrics['tool_acc'] / total['tool'] * 100,
                arg_acc=metrics['arg_acc'] / total['tool'] * 100,
                answer_acc=metrics['answer_acc'] / total['answer'] * 100,
                tool_call=metrics['tool_call'],
                tool_call_error=metrics['tool_call_error']
            )
        elif self.mode == 'every':
            total = {'all': 0, 'answer': 0, 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            total_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            precision = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            recall = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            f1 = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            metrics = {
                'pass_rate': 0, 'answer_acc': 0, 'tool_call': 0, 'tool_call_error': 0
            }

            for preds, gts, ref in zip(predictions, gold, references):
                ref = json.loads(ref)
                total['all'] += 1
                if ref:
                    total['answer'] += 1
                pred_type, pred_answer = self.get_response_type(preds[0][-1])
                if pred_type == 'answer' and pred_answer and ref:
                    if isinstance(ref, dict) and pred_answer:
                        metrics['answer_acc'] += self.iscorrect(pred_answer, ref)
                    elif isinstance(ref, list) and pred_answer:
                        metrics['answer_acc'] += self.simscore(pred_answer, ref)
                error_flag = 0
                tool_flag = 0
                for pred in preds[0]:
                    if 'tool_calls' in pred:
                        tool_flag = 1
                        metrics['tool_call'] += 1
                        if 'error' in pred:
                            metrics['tool_call_error'] += 1
                    if 'error' in pred:
                        error_flag = 1                       
                if pred_type == 'answer' and pred_answer and not error_flag and tool_flag:
                    metrics['pass_rate'] += 1
                
                pred_tool_calls = []
                for pred in preds[0]:
                    if 'tool_calls' in pred:
                        tool_type = self.gettype(pred['tool_calls'][0]['function']['name'])
                        if tool_type in total_predict:
                            total_predict[tool_type] += 1
                        pred_tool_calls.append(pred['tool_calls'][0]['function']['name'])
                for gt in gts[0]:
                    if 'tool_calls' in gt:
                        tool_type = self.gettype(gt['tool_calls'][0]['function']['name'])
                        total[tool_type] += 1
                        if gt['tool_calls'][0]['function']['name'] in pred_tool_calls:
                            metrics[tool_type] += 1
            for tool_type in f1.keys():
                precision[tool_type] = metrics[tool_type] / (total_predict[tool_type] + 1e-5)
                recall[tool_type] = metrics[tool_type] / total[tool_type]
                f1[tool_type] = 2 * precision[tool_type] * recall[tool_type] / (precision[tool_type] + recall[tool_type] + 1e-5)
            return dict(
                pass_rate=metrics['pass_rate'] / total['all'] * 100,
                answer_acc=metrics['answer_acc'] / total['answer'] * 100,
                tool_call=metrics['tool_call'],
                tool_call_error=metrics['tool_call_error'],
                p_f1 = f1['perception'] * 100,
                o_f1 = f1['operation'] * 100,
                l_f1 = f1['logic'] * 100,
                c_f1 = f1['creativity'] * 100
            )
        else:
            raise NotImplementedError
