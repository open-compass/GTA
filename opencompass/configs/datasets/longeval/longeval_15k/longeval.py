from mmengine.config import read_base

with read_base():
    from .longeval_2wikimqa_e.longeval_2wikimqa_e_gen_6b3efc import longeval_2wikimqa_e_datasets
    from .longeval_hotpotqa_e.longeval_hotpotqa_e_gen_6b3efc import longeval_hotpotqa_e_datasets
    from .longeval_lines.longeval_lines_gen_7daf4a import longeval_lines_datasets
    from .longeval_stackselect.longeval_stackselect_gen_c92c67 import longeval_stackselect_datasets
    from .longeval_textsort.longeval_textsort_gen_4d7bcb import longeval_textsort_datasets
    from .longeval_trec_e.longeval_trec_e_gen_824187 import longeval_trec_e_datasets

longeval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
