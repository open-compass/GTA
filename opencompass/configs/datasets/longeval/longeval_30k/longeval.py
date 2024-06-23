from mmengine.config import read_base

with read_base():
    from .longeval_lines.longeval_lines_gen_70f645 import longeval_lines_datasets
    from .longeval_stackselect.longeval_stackselect_gen_42fd23 import longeval_stackselect_datasets
    from .longeval_textsort.longeval_textsort_gen_781c84 import longeval_textsort_datasets

longeval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
