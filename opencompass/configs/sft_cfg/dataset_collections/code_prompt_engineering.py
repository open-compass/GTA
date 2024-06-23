from mmengine.config import read_base

with read_base():
    # the new prompt can get better performance than the old one
    from ...datasets.mbpp.mbpp_gen_caa7ab import mbpp_datasets
    from ...datasets.humaneval.humaneval_gen_6d1cc2 import humaneval_datasets

    # old prompt setting
    # from ...datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
