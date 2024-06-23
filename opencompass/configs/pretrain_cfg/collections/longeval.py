from mmengine.config import read_base

with read_base():
    from ...datasets.longeval.longeval_30k.longeval import longeval_datasets as longeval_30k_datasets
    from ...datasets.longeval.longeval_15k.longeval import longeval_datasets as longeval_15k_datasets
    from ...datasets.longeval.longeval_8k.longeval import longeval_datasets as longeval_8k_datasets
    from ...datasets.longeval.longeval_2k.longeval import longeval_datasets as longeval_2k_datasets
    from ...datasets.longeval.longeval_4k.longeval import longeval_datasets as longeval_4k_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
