from mmengine.config import read_base

with read_base():
    from .longeval_2k.longeval import longeval_datasets as longeval_2k_datasets
    from .longeval_4k.longeval import longeval_datasets as longeval_4k_datasets
    from .longeval_8k.longeval import longeval_datasets as longeval_8k_datasets
    from .longeval_15k.longeval import longeval_datasets as longeval_15k_datasets
    from .longeval_30k.longeval import longeval_datasets as longeval_30k_datasets

longeval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
