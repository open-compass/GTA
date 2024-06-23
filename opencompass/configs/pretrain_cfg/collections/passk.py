from mmengine.config import read_base
from opencompass.datasets import MBPPDataset_V2, MBPPPassKEvaluator

with read_base():
    from ...datasets.humaneval.humaneval_gen_4a6eef import humaneval_datasets
    from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets

n = 30
k = [1, 5, 10]

humaneval_datasets[0]['abbr'] = f'openai_humaneval_repeat{n}'
humaneval_datasets[0]['num_repeats'] = n
mbpp_datasets[0]['abbr'] = f'mbpp_repeat{n}'
mbpp_datasets[0]['num_repeats'] = n
mbpp_datasets[0]['type'] = MBPPDataset_V2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'
humaneval_datasets[0]['eval_cfg']['evaluator']['k'] = k
mbpp_datasets[0]['eval_cfg']['evaluator']['k'] = k


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
