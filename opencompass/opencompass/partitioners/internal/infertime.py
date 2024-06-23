import copy
import math
import os.path as osp
from typing import Dict, List, Optional

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path)

from ..base import BasePartitioner


@PARTITIONERS.register_module()
class InferTimePartitioner(BasePartitioner):
    """This Inferencer uses the results of time consumption on a common dataset
    list with an InternLMwithModule 7B Base and an InternLMwithModule 70B Chat
    model as the basis. According to these time consumption results, the
    Inferencer splits or merges different task processes, forming multiple
    jobs, and ensures that each job's duration does not exceed a certain limit.
    We believe that different models exhibit similar time consumption trends on
    the same dataset, meaning if a dataset is slow on Model A, it will also be
    slow on Model B. We hope to shorten the total evaluation time through this
    method.

    Due to the above implementation mechanism, this Inferencer has the
    following issues that need special attention:

    * The same dataset, different prompt words, and different evaluation
      methods (ppl/gen) can cause significant variations in evaluation time.
      However, our evaluation speed is based on `dataset_abbr`, without
      considering the aforementioned issues.
    * Currently, the computation of time consumption is incomplete. New
      datasets cannot directly estimate evaluation time. By default, no
      splitting or merging is done for datasets not listed in
      `infer_time_path`. After the evaluation is completed, there will also be
      no updates to `infer_time_path`.
    * The unit of our `max_task_time` can correspond to seconds to some extent,
      but there is an underlying proportional relationship involved, which does
      not allow for direct equivalence.


    Args:
        out_dir (str): The output directory of tasks.
        max_task_size (int): The maximum size of a task.
        strategy (str): The partition strategy. Supported strategies are:
            'heuristic' and 'split'. Defaults to 'heuristic'.
            heuristic: split large datasets into several tasks, merge small
                datasets into one task.
            split: split large datasets into several tasks only.
        infer_time_path (str): The path to the infer time record file.
        dataset_size_path (str): The path to the dataset size cache file.
        keep_keys (list[str]): The keys to be kept from the experiment config
            to the task config.
    """

    DEFAULT_INFER_TIME_PATH = osp.join(osp.dirname(__file__), 'infertime.json')

    def __init__(self,
                 out_dir: str,
                 max_task_time: int = 3600,
                 strategy: str = 'heuristic',
                 infer_time_path: str = DEFAULT_INFER_TIME_PATH,
                 dataset_size_path: str = '.cache/dataset_size.json',
                 keep_keys: Optional[List[str]] = None):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)

        self.max_task_time = max_task_time
        self.infer_time_path = infer_time_path
        self.dataset_size_path = dataset_size_path
        assert strategy in ('heuristic', 'split'), \
            f'Unsupported partition strategy: {strategy}. '\
            'Supported strategies are: `heuristic`, `split` .'
        self.strategy = strategy

    def partition(self,
                  model_dataset_combinations: List[Dict[str, List]],
                  work_dir: str,
                  out_dir: str,
                  add_cfg: Dict = {}) -> List[ConfigDict]:

        # intentionally avoid any sort here,
        # for user's abaility to manipulate the order
        tasks = []
        for comb in model_dataset_combinations:
            for model in comb['models']:
                chunks = []  # elements: tuple(size, dataset_chunk)
                for dataset in comb['datasets']:
                    filename = get_infer_output_path(model, dataset, out_dir)
                    # skip the task if the task output exists
                    if osp.exists(filename):
                        continue
                    infer_time = self.get_cost(dataset)
                    if infer_time > self.max_task_time:
                        root, ext = osp.splitext(filename)
                        dataset_splits = self.split_dataset(dataset)
                        for i, dataset_split in enumerate(dataset_splits):
                            if not osp.exists(f'{root}_{i}{ext}'):
                                chunks.append(
                                    (self.max_task_time, dataset_split))
                    else:
                        chunks.append((infer_time, dataset))

                if self.strategy == 'heuristic':
                    current_size, current_chunks = 0, []
                    for index in range(len(chunks)):
                        current_size += chunks[index][0]
                        current_chunks.append(chunks[index][1])
                        if index == len(chunks) - 1 or current_size + chunks[
                                index + 1][0] > self.max_task_time:
                            tasks.append(
                                Config({
                                    'models': [model],
                                    'datasets': [current_chunks],
                                    'work_dir': work_dir,
                                    **add_cfg
                                }))
                            current_size, current_chunks = 0, []
                elif self.strategy == 'split':
                    for _, dataset in chunks:
                        tasks.append(
                            Config({
                                'models': [model],
                                'datasets': [[dataset]],
                                'work_dir': work_dir,
                                **add_cfg
                            }))

        return tasks

    @property
    def infer_time(self):
        if not hasattr(self, '_infer_time'):
            if osp.exists(self.infer_time_path):
                self._infer_time = mmengine.load(self.infer_time_path)
            else:
                self._infer_time = {}
        return self._infer_time

    @property
    def dataset_size(self):
        if not hasattr(self, '_dataset_size'):
            if osp.exists(self.dataset_size_path):
                self._dataset_size = mmengine.load(self.dataset_size_path)
            else:
                self._dataset_size = {}
        return self._dataset_size

    def split_dataset(self, dataset_cfg: ConfigDict) -> List[ConfigDict]:
        """Split dataset into several parts."""
        infer_time = self.get_cost(dataset_cfg)
        dataset_size = self.get_size(dataset_cfg)
        split_configs = []
        abbr = dataset_abbr_from_cfg(dataset_cfg)
        # evenly distribute the task
        num_split = math.ceil(infer_time / self.max_task_time)
        step = math.ceil(dataset_size / num_split)
        for part, i in enumerate(range(0, dataset_size, step)):
            cfg = copy.deepcopy(dataset_cfg)
            cfg['abbr'] = abbr + f'_{part}'
            test_range = cfg['reader_cfg'].get('test_range', '')
            cfg['reader_cfg']['test_range'] = f'{test_range}[{i}:{i+step}]'
            split_configs.append(cfg)
        return split_configs

    def get_cost(self, dataset: ConfigDict) -> float:
        dataset_abbr = dataset_abbr_from_cfg(dataset)

        if dataset_abbr in self.infer_time:
            return self.infer_time[dataset_abbr]
        else:
            return self.max_task_time

    def get_size(self, dataset: ConfigDict) -> int:
        dataset_abbr = dataset_abbr_from_cfg(dataset)

        test_range = dataset.reader_cfg.get('test_range', '')

        if dataset_abbr in self.dataset_size:
            actual_size = eval('len(range(self.dataset_size[dataset_abbr])'
                               f'{test_range})')
            return actual_size

        dataset = build_dataset_from_cfg(dataset)
        self.dataset_size[dataset_abbr] = len(dataset.test)

        mmengine.mkdir_or_exist('.cache/')
        mmengine.dump(self.dataset_size,
                      self.dataset_size_path,
                      indent=4,
                      ensure_ascii=False)

        actual_size = eval('len(range(self.dataset_size[dataset_abbr])'
                           f'{test_range})')
        return actual_size
