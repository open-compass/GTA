
from mmengine.config import read_base

with read_base():
    from ..humaneval.humaneval_gen_8e312c import humaneval_datasets
    from ..humaneval_cn.humaneval_cn_gen_6313aa import humaneval_cn_datasets
    from ..humanevalx.humanevalx_gen_620cfa import humanevalx_datasets
    from ..humaneval_plus.humaneval_plus_gen_8e312c import humaneval_plus_datasets
    from ..mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from ..mbpp_cn.mbpp_cn_gen_1d1481 import mbpp_cn_datasets
    from ..mbpp.sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets

    from ..humaneval.humaneval_passk_gen_8e312c import humaneval_datasets as humaneval_passk_datasets
    from ..humaneval_cn.humaneval_cn_passk_gen_6313aa import humaneval_cn_datasets as humaneval_cn_passk_datasets
    from ..humaneval_plus.humaneval_plus_passk_gen_8e312c import humaneval_plus_datasets as humaneval_plus_passk_datasets
    from ..mbpp.mbpp_passk_gen_1e1056 import mbpp_datasets as mbpp_passk_datasets
    from ..mbpp_cn.mbpp_cn_passk_gen_1d1481 import mbpp_cn_datasets as mbpp_cn_passk_datasets
    from ..mbpp.sanitized_mbpp_passk_gen_1e1056 import sanitized_mbpp_datasets as sanitized_mbpp_passk_datasets

    from ..humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets as humaneval_repeat10_datasets
    from ..humaneval_cn.humaneval_cn_repeat10_gen_6313aa import humaneval_cn_datasets as humaneval_cn_repeat10_datasets
    from ..humaneval_plus.humaneval_plus_repeat10_gen_8e312c import humaneval_plus_datasets as humaneval_plus_repeat10_datasets
    from ..mbpp.mbpp_repeat10_gen_1e1056 import mbpp_datasets as mbpp_repeat10_datasets
    from ..mbpp_cn.mbpp_cn_repeat10_gen_1d1481 import mbpp_cn_datasets as mbpp_cn_repeat10_datasets
    from ..mbpp.sanitized_mbpp_repeat10_gen_1e1056 import sanitized_mbpp_datasets as sanitized_mbpp_repeat10_datasets

compassbench_v1_code_datasets = sum([v for k, v in locals().items() if (k.endswith("_datasets") and not k.endswith("_passk_datasets") and not k.endswith("_repeat10_datasets"))], [])
compassbench_v1_code_passk_datasets = sum([v for k, v in locals().items() if k.endswith("_passk_datasets")], [])
compassbench_v1_code_repeat10_datasets = sum([v for k, v in locals().items() if k.endswith("_repeat10_datasets")], [])
