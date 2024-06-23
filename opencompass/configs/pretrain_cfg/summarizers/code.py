from mmengine.config import read_base
from opencompass.summarizers import PretrainSummarizer

with read_base():
    from .groups.pretrain import  coding_summary_groups
    from .groups.humanevalx import humanevalx_summary_groups
    from .groups.ds1000 import ds1000_summary_groups

summarizer = dict(
    type=PretrainSummarizer,
    dataset_abbrs=[
        # '--- Coding ---',
        'openai_humaneval',
        'mbpp',
        'coding-average',
        'maxmin',
        'humanevalx',
        'ds1000',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
