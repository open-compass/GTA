from mmengine.config import read_base
from opencompass.datasets import MBPPDataset_V2, MBPPPassKEvaluator
with read_base():
    from ...datasets.humanevalx.humanevalx_gen_620cfa import humanevalx_datasets
    # from ...datasets.py150.py150_gen import py150_datasets
    from ...datasets.clozeTest_maxmin.clozeTest_maxmin_gen import maxmin_datasets
    from ...datasets.ds1000.ds1000_compl_service_eval_gen_cbc84f import ds1000_datasets

    from ...datasets.humaneval.humaneval_gen_4a6eef import humaneval_datasets
    from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
