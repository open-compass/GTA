import base64
import getpass
import json
import os
import os.path as osp
import time
import uuid
from datetime import datetime
from typing import Dict, Union
import math

import mmengine
import openxlab.xlab
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from mmengine.config import Config
from mmengine.utils import track_parallel_progress
from tqdm import tqdm

from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg
import opencompass

METRIC_WHITE_LIST = [
    'accuracy', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'score',
    'humaneval_pass@1', 'auc_score', 'avg_toxicity_score', 'bleurt_diff',
    'matthews_correlation', 'truth', 'info', 'exact_match', 'f1'
]
# WARNING!!!
# DO NOT CHANGE THIS KEY!!!
# 由于面向实验室内部多个团队使用，为了将来取数据的方便，这个 key 需统一使用且不能改变
INTERNAL_USAGE_KEY = b'InternalUsageKey'
OPENCOMPASS_VERSION = opencompass.__version__
JWT = openxlab.xlab.handler.user_token.get_jwt()
UID = openxlab.xlab.handler.user_token.get_token().sso_uid

def encrypt_number(number, key=INTERNAL_USAGE_KEY):
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(str(number).encode(), AES.block_size))
    return base64.b64encode(encrypted).decode()


class UploadedResultsCache:
    def __init__(self, uploaded_result_path: str, dry_run: bool = False):
        self.uploaded_result_path = uploaded_result_path
        self.dry_run = dry_run
        print("Uploaded cache results path:", self.uploaded_result_path)

    @property
    def uploaded_results(self):
        if not hasattr(self, '_uploaded_results'):
            if osp.exists(self.uploaded_result_path):
                self._uploaded_results = mmengine.load(
                    self.uploaded_result_path)
            else:
                self._uploaded_results = {}
        return self._uploaded_results

    @property
    def model_ids(self):
        if 'model_ids' not in self.uploaded_results:
            self.uploaded_results['model_ids'] = []
        return self.uploaded_results['model_ids']

    @property
    def dataset_and_prompt_ids(self):
        if 'dataset_and_prompt_ids' not in self.uploaded_results:
            self.uploaded_results['dataset_and_prompt_ids'] = []
        return self.uploaded_results['dataset_and_prompt_ids']

    @property
    def record_ids(self):
        if 'record_ids' not in self.uploaded_results:
            self.uploaded_results['record_ids'] = []
        return self.uploaded_results['record_ids']

    @property
    def dataset_item_ids(self):
        if 'dataset_item_ids' not in self.uploaded_results:
            self.uploaded_results['dataset_item_ids'] = []
        return self.uploaded_results['dataset_item_ids']

    @property
    def eval_item_ids(self):
        if 'eval_item_ids' not in self.uploaded_results:
            self.uploaded_results['eval_item_ids'] = []
        return self.uploaded_results['eval_item_ids']

    @property
    def answer_dataset_id_mapping(self):
        if 'answer_dataset_id_mapping' not in self.uploaded_results:
            self.uploaded_results['answer_dataset_id_mapping'] = {}
        return self.uploaded_results['answer_dataset_id_mapping']

    def save(self):
        if self.dry_run:
            return
        mmengine.dump(self.uploaded_results, self.uploaded_result_path, indent=4, ensure_ascii=False)


def get_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, 'm' + s))

def convert_to_timestamp(date_str, date_format='%Y%m%d_%H%M%S') -> str:
    dt = datetime.strptime(date_str, date_format)
    timestamp = dt.timestamp()
    return str(int(timestamp))


class DatabaseReporter:

    def __init__(
            self,
            work_dir: str,
            uploaded_result_path: str = None,
            url: str = "http://inferstore-eval.openxlab.org.cn",
            retry: int = 2,
            nproc: int = 32,
            debug: bool = False,
            dry_run: bool = False,
            config_name: str = None,
        ):

        if uploaded_result_path is None:
            uploaded_result_path = os.path.join(os.path.expanduser('~'), '.cache', 'opencompass', 'results-upload-database', os.path.basename(osp.normpath(work_dir)) + '.json')
        os.makedirs(os.path.dirname(uploaded_result_path), exist_ok=True)
        self.work_dir = work_dir
        self.url = url  # TODO: Add this to secret
        self.eval_toolkit_version = OPENCOMPASS_VERSION
        self.eval_person = getpass.getuser()  # Get the username of the current user
        self.dataset_size_path = '.cache/dataset_size.json'
        self.uploaded_results = UploadedResultsCache(uploaded_result_path)
        self.retry = retry
        self.nproc = nproc
        self.debug = debug
        self.dry_run = dry_run
        self.config_name = config_name

    def get_dataset_type(self, dataset_cfg):
        if 'PPLInferencer' in dataset_cfg['infer_cfg']['inferencer']['type']:
            dataset_type = 'discriminative'
        elif 'GenInferencer' in dataset_cfg['infer_cfg']['inferencer']['type']:
            dataset_type = 'generate'
        else:
            dataset_type = 'unknown'
        return dataset_type

    @property
    def dataset_size(self):
        if not hasattr(self, '_dataset_size'):
            if osp.exists(self.dataset_size_path):
                self._dataset_size = mmengine.load(self.dataset_size_path)
            else:
                self._dataset_size = {}
        return self._dataset_size

    def resolve(self, model_cfg, dataset_cfg):
        infos = {}
        time_fmt = osp.basename(osp.normpath(self.work_dir))
        if model_cfg is not None:
            infos["model_name"] = model_abbr_from_cfg(model_cfg)
            infos["model_version"] = "0.0.2" # do not change this magic
            infos["model_prompt"] = json.dumps(model_cfg.get('meta_template', {}), sort_keys=True)
            infos["model_id"] = get_uuid(infos["model_name"] + '--' + infos["model_version"] + '--' + infos["model_prompt"] + '--' + time_fmt)
        if dataset_cfg is not None:
            infos['dataset_prompt'] = json.dumps(dataset_cfg.to_dict(), sort_keys=True)
            infos["dataset_prompt_id"] = get_uuid(infos['dataset_prompt'])
            infos["dataset_name"] = dataset_abbr_from_cfg(dataset_cfg)
            infos["dataset_type"] = self.get_dataset_type(dataset_cfg)
            infos["dataset_version"] = "0.0.1"
            infos["dataset_item_cnt"] = self.dataset_size.get(infos["dataset_name"], 1)
            infos["dataset_tags"] = [infos["dataset_name"]]
            infos["dataset_id"] = get_uuid(infos["dataset_name"] + '--' + infos["dataset_version"] + '--' + str(infos["dataset_item_cnt"]) + '--' + str(infos["dataset_tags"]))
            infos["dataset_and_prompt_id"] = infos["dataset_id"] + '--' + infos["dataset_prompt_id"]
        infos["eval_toolkit_version"] = self.eval_toolkit_version
        infos["eval_start_ts"] = convert_to_timestamp(time_fmt)
        infos["eval_end_ts"] = str(int(datetime.now().timestamp()))
        infos["eval_person"] = self.eval_person
        if model_cfg is not None and dataset_cfg is not None:
            infos["eval_task_id"] = get_uuid(infos["eval_start_ts"] + '--' + infos["eval_person"] + '--' + infos["eval_toolkit_version"] + '--' + infos["model_id"])
            infos["record_id"] = get_uuid(infos["eval_task_id"] + '--' + infos["dataset_and_prompt_id"])
            key = f'{infos["dataset_id"]}--{infos["dataset_prompt_id"]}--{infos["model_id"]}--{infos["dataset_item_cnt"]}'
            if key in self.uploaded_results.answer_dataset_id_mapping:
                infos["answer_dataset_id"] = self.uploaded_results.answer_dataset_id_mapping[key]
            else:
                infos["answer_dataset_id"] = None
        return infos

    def create_model(self, model_cfg):
        infos =  self.resolve(model_cfg, None)
        if infos["model_id"] not in self.uploaded_results.model_ids:
            content = {
                'model_id': infos["model_id"],
                'model_name': infos["model_name"],
                'model_version': infos["model_version"],
                'model_prompt': infos["model_prompt"]
            }
            success, ret = self.post('CreateModel', content)
            if not success:
                return ret
        return infos["model_id"]

    def create_dataset(self, dataset_cfg):
        infos = self.resolve(None, dataset_cfg)
        if infos["dataset_and_prompt_id"] not in self.uploaded_results.dataset_and_prompt_ids:
            content = {
                'dataset_prompt_id': infos["dataset_prompt_id"],
                'dataset_prompt': infos["dataset_prompt"],
            }
            # print(content)
            success, ret = self.post('CreateDatasetPrompt', content)
            # print(success, ret)
            if not success:
                return ret

            content = {
                'dataset_id': infos["dataset_id"],
                'dataset_name': infos["dataset_name"],
                'dataset_type': infos["dataset_type"],
                'dataset_version': infos["dataset_version"],
                'dataset_item_cnt': infos["dataset_item_cnt"],
                'dataset_tags': infos["dataset_tags"],
                'dataset_prompt': [infos["dataset_prompt_id"]],
            }
            success, ret = self.post('CreateEvalDataset', content)
            if not success:
                return ret

            content = {
                'dataset_id': infos["dataset_id"],
                'dataset_prompt': [infos["dataset_prompt_id"]],
            }
            success, ret = self.post('AddDatasetPrompt', content)
            if not success:
                return ret

        return infos["dataset_and_prompt_id"]

    def create_obj_eval_task(self, task):
        model_cfg, dataset_cfg = task
        infos = self.resolve(model_cfg, dataset_cfg)
        result_file = osp.join(self.work_dir, 'results', infos["model_name"], infos["dataset_name"] + '.json')
        if not osp.exists(result_file):
            return
        with open(result_file, 'r') as f:
            eval_scores = json.load(f)
        eval_scores = {k: encrypt_number(round(float(v), 4)) for k, v in eval_scores.items() if k in METRIC_WHITE_LIST}
        if infos["answer_dataset_id"] is None:
            return ValueError(f'answer dataset id not found for {infos["dataset_id"]}--{infos["dataset_prompt_id"]}--{infos["model_id"]}--{infos["dataset_item_cnt"]}')

        if infos["record_id"] not in self.uploaded_results.record_ids:
            content = {
                'eval_task_id': infos["eval_task_id"],
                'model_id': infos["model_id"],
                'question_dataset_id': infos["dataset_id"],
                'answer_dataset_id': infos["answer_dataset_id"],
                'dataset_prompt_id': infos["dataset_prompt_id"],
                'eval_toolkit_version': infos["eval_toolkit_version"],
                'eval_start_ts': infos["eval_start_ts"],
                'eval_end_ts': infos["eval_end_ts"],
                'eval_person': infos["eval_person"],
                'eval_scores': eval_scores,
            }
            success, ret = self.post('CreateObjEvalTask', content)
            if not success:
                return ret
        return infos["record_id"]

    def _insert_dataset_item(self, dataset_cfg):
        from opencompass.utils import build_dataset_from_cfg
        dataset = build_dataset_from_cfg(dataset_cfg)
        infos = self.resolve(None, dataset_cfg)

        if infos["dataset_id"] not in self.uploaded_results.dataset_item_ids:
            for offset, line in enumerate(tqdm(dataset.test)):
                input = json.dumps({k: line.get(k, '-') for k in dataset.reader.input_columns})
                reference = json.dumps(line[dataset.reader.output_column])
                content = {
                    'dataset_id': infos["dataset_id"],
                    'tag': infos["dataset_tags"],
                    'offset': offset,
                    'input': input,
                    'reference': reference,
                }
                success, ret = self.post('InsertDatasetItem', content)
                if not success:
                    return ret
        return infos["dataset_id"]

    def insert_dataset_item(self, dataset_cfg):
        try:
            return self._insert_dataset_item(dataset_cfg)
        except Exception as e:
            return e

    def make_answer_dataset_id_mapping(self, task):
        # this should run in serial

        model_cfg, dataset_cfg = task
        infos = self.resolve(model_cfg, dataset_cfg)

        key = f'{infos["dataset_id"]}--{infos["dataset_prompt_id"]}--{infos["model_id"]}--{infos["dataset_item_cnt"]}'
        if key not in self.uploaded_results.answer_dataset_id_mapping:
            content = {
                'dataset_id': infos["dataset_id"],
                'prompt_id': infos["dataset_prompt_id"],
                'model_id': infos["model_id"],
                'item_cnt': infos["dataset_item_cnt"],
            }
            success, ret = self.post('CreateEvalAnswerSet', content)
            if not success:
                return ret

            # extract answer dataset id
            if ret.get('errCode', 0) == -1:
                answer_dataset_id = ret['errMsg'].lower().split('duplicated answer set: ')[-1].split('.')[0]
            else:
                answer_dataset_id = ret['answer set id']
        else:
            answer_dataset_id = self.uploaded_results.answer_dataset_id_mapping[key]
        return key + '@' + answer_dataset_id

    def need_to_run_insert_eval_answer_and_result_item(self, task):
        # this should run in serial
        model_cfg, dataset_cfg = task
        infos = self.resolve(model_cfg, dataset_cfg)
        if infos["record_id"] not in self.uploaded_results.eval_item_ids:
            return True
        return False

    def insert_eval_answer_and_result_item(self, task):
        # this should run in serial
        model_cfg, dataset_cfg = task
        infos = self.resolve(model_cfg, dataset_cfg)
        if infos["record_id"] not in self.uploaded_results.eval_item_ids:
            preds = self.load_preds(model_cfg, dataset_cfg)
            if preds is None:
                return ValueError('preds not found')
            # upload predictions one by one
            for offset in tqdm(range(len(preds))):
                # formulate
                ppl_scores = {}
                if infos["dataset_type"] == 'discriminative':
                    for key in preds[str(offset)].keys():
                        if key in ['in-context examples', 'prediction']:
                            continue
                        if key.startswith('label: '):
                            key = key[7:]
                            ppl_scores[key] = preds[str(offset)]['label: ' + key]['PPL']
                output = str(preds[str(offset)]['prediction'])
                if not output:
                    output = '[None]'
                ppl_scores = {k: (v if not math.isnan(v) else -1) for k, v in ppl_scores.items()}
                # upload
                content = {
                    'answerset_id': infos["answer_dataset_id"],
                    'offset': offset,
                    'output': output,
                    'ppl_scores': ppl_scores,
                    'eval_result': {},
                }
                success, ret = self.post('InsertEvalAnswerAndResItem', content)
                if not success:
                    return ret
        return infos["record_id"]

    def load_preds(self, model_cfg, dataset_cfg):
        model_abbr = model_abbr_from_cfg(model_cfg)
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
        filename = osp.join(self.work_dir, 'predictions', model_abbr, dataset_abbr + '.json')
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext
        if not osp.exists(osp.realpath(filename)) and not osp.exists(osp.realpath(partial_filename)):
            return None
        if osp.exists(osp.realpath(filename)):
            preds = mmengine.load(filename)
        else:
            filename = partial_filename
            preds, offset = {}, 0
            i = 1
            while osp.exists(osp.realpath(filename)):
                _preds = mmengine.load(filename)
                filename = root + f'_{i}' + ext
                i += 1
                for _o in range(len(_preds)):
                    preds[str(offset)] = _preds[str(_o)]
                    offset += 1
        return preds

    def _launch(self, func, inputs, nproc=None, keep_order=False):
        if not self.debug:
            if nproc is None:
                nproc = self.nproc
            return track_parallel_progress(func, inputs, nproc=nproc, keep_order=keep_order)
        else:
            return [func(i) for i in tqdm(inputs)]

    def _load_cfg(self):
        # get lateset configs
        if self.config_name is not None:
            config_name = self.config_name
            if not config_name.endswith('.py'):
                config_name = config_name + '.py'
        else:
            config_name = sorted(os.listdir(osp.join(self.work_dir, 'configs')))[-1]
        cfg = Config.fromfile(osp.join(self.work_dir, 'configs', config_name), format_python_code=False)
        return cfg


    def run_meta(self):
        cfg = self._load_cfg()
        timestamp = osp.basename(osp.normpath(self.work_dir))
        error = False

        # create model
        model_ids = self._launch(self.create_model, cfg['models'])
        for model_id in model_ids:
            if isinstance(model_id, str):
                if model_id not in self.uploaded_results.model_ids:
                    self.uploaded_results.model_ids.append(model_id)
            else:
                print(f'something wrong when creating model: {str(model_id)}')
                error = True
        self.uploaded_results.save()
        if error:
            return

        # create dataset
        dataset_and_prompt_ids = self._launch(self.create_dataset, cfg['datasets'])
        for dataset_and_prompt_id in dataset_and_prompt_ids:
            if isinstance(dataset_and_prompt_id, str):
                if dataset_and_prompt_id not in self.uploaded_results.dataset_and_prompt_ids:
                    self.uploaded_results.dataset_and_prompt_ids.append(
                        dataset_and_prompt_id)
            else:
                print(f'something wrong when creating dataset: {str(dataset_and_prompt_id)}')
                error = True
        self.uploaded_results.save()
        if error:
            return

    def run_answer_dataset_id_mapping(self):
        cfg = self._load_cfg()
        timestamp = osp.basename(osp.normpath(self.work_dir))

        model_ids = self._launch(self.create_model, cfg['models'], keep_order=True)
        dataset_and_prompt_ids = self._launch(self.create_dataset, cfg['datasets'], keep_order=True)

        # create preds
        tasks = []
        for model_cfg in cfg['models']:
            for dataset_cfg in cfg['datasets']:
                tasks.append((model_cfg, dataset_cfg))

        rets = self._launch(self.make_answer_dataset_id_mapping, tasks)
        for ret in rets:
            if isinstance(ret, str):
                key, answer_dataset_id = ret.split('@')
                self.uploaded_results.answer_dataset_id_mapping[key] = answer_dataset_id
            else:
                print(f'something wrong when creating answer set mapping: {str(ret)}')
        self.uploaded_results.save()

    def run_result(self):
        cfg = self._load_cfg()
        timestamp = osp.basename(osp.normpath(self.work_dir))
        error = False

        # create score
        tasks = []
        for model_cfg in cfg['models']:
            for dataset_cfg in cfg['datasets']:
                tasks.append((model_cfg, dataset_cfg))

        record_ids = self._launch(self.create_obj_eval_task, tasks)
        for record_id in record_ids:
            if isinstance(record_id, str):
                if record_id not in self.uploaded_results.record_ids:
                    self.uploaded_results.record_ids.append(record_id)
            elif record_id is None:  # results doesn't exist
                continue
            else:
                print(f'something wrong when creating record: {str(record_id)}')
                error = True
        self.uploaded_results.save()
        if error:
            return

    def run_raw_dataset(self):
        cfg = self._load_cfg()

        dataset_item_ids = self._launch(self.insert_dataset_item, cfg['datasets'])
        for dataset_item_id in dataset_item_ids:
            if isinstance(dataset_item_id, str):
                if dataset_item_id not in self.uploaded_results.dataset_item_ids:
                    self.uploaded_results.dataset_item_ids.append(dataset_item_id)
            else:
                print(f'something wrong when creating dataset: {str(dataset_item_id)}')
        self.uploaded_results.save()

    def run_prediction(self):
        cfg = self._load_cfg()
        _ = self._launch(self.create_model, cfg['models'])
        _ = self._launch(self.create_dataset, cfg['datasets'])

        # create preds
        tasks = []
        for dataset_cfg in cfg['datasets']:
            for model_cfg in cfg['models']:
                task = (model_cfg, dataset_cfg)
                if self.need_to_run_insert_eval_answer_and_result_item(task):
                    tasks.append(task)

        batch_size = 256
        for index in range(0, len(tasks), batch_size):
            subtasks = tasks[index:index + batch_size]
            eval_item_ids = self._launch(self.insert_eval_answer_and_result_item, subtasks)
            for eval_item_id in eval_item_ids:
                if isinstance(eval_item_id, str):
                    if eval_item_id not in self.uploaded_results.eval_item_ids:
                        self.uploaded_results.eval_item_ids.append(eval_item_id)
                else:
                    print(f'something wrong when creating dataset: {str(eval_item_id)}')
            self.uploaded_results.save()

    def post(self, endpoint: str, content: Union[str, Dict]) -> (bool, Union[Exception, Dict]):
        """Post a message to endpoint.

        若成功，返回 (True, response (dict))
        若失败，返回 (False, error)
        """
        if self.dry_run:
            print(f'POST {endpoint} {content}')
            return True, {}
        elif self.debug:
            print(f'POST {endpoint} {content}')

        header = {
            'Content-Type': 'application/json',
            "authorization": JWT,
            'id': UID,
        }
        url = osp.join(self.url, endpoint)
        retry = self.retry
        if self.debug:
            print(f'url = "{url}"')
            print(f'data = {content}')
            print(f'headers = {header}')
        if isinstance(content, dict):
            content = json.dumps(content)
        while True:
            try:
                r = requests.post(url, data=content, headers=header)
                break
            except Exception as e:
                retry -= 1
                if retry <= 0:
                    print(f'Error when posting to {url}: {e}\nContent: {content}')
                    return False, e
                else:
                    time.sleep(1)
        resp_content = r.content.decode('utf-8')
        if r.status_code != 200:
            if 'duplicate' not in resp_content.lower():
                # better format
                try:
                    resp_content = json.dumps(json.loads(resp_content))
                except Exception:
                    pass
                print(f'Error when posting to {url}: {resp_content}\nContent: {content}')
                return False, Exception(resp_content)
            else:
                if self.debug:
                    print(f'Error when posting to {url}: {resp_content}\nContent: {content}')
                return True, json.loads(r.content.decode('utf-8'))
        else:
            if self.debug:
                print(resp_content)
        return True, json.loads(r.content.decode('utf-8'))
