from mmengine.config import read_base

with read_base():
    from ...datasets.leval.leval import leval_datasets
    from ...datasets.longbench.longbench import longbench_datasets
    from ...datasets.longeval.longeval import longeval_datasets

LONGTEXT_DATASET_NAMES = [
    'longeval_datasets', 
    'longbench_datasets', 
    'leval_datasets'
]
BLACKLIST_DATASET_NAMES = [
]

longtext_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in LONGTEXT_DATASET_NAMES and k not in BLACKLIST_DATASET_NAMES), [])
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
