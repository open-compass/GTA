from mmengine.config import read_base

with read_base():
    from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from ...datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from ...datasets.ds1000.ds1000_service_eval_gen_cbc84f import ds1000_datasets
    from ...datasets.clozeTest_maxmin.clozeTest_maxmin_gen_c205fb import maxmin_datasets
    from ...datasets.py150.py150_gen_38b13d import py150_datasets
    # humanevalx need docker during eval, not fully support now
    # from ...datasets.humanevalx.humanevalx_gen_0af626 import humanevalx_datasets
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
