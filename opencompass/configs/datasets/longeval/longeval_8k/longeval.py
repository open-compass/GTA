from mmengine.config import read_base

with read_base():
    from .longeval_2wikimqa_e.longeval_2wikimqa_e_gen_6b3efc import longeval_2wikimqa_e_datasets
    from .longeval_gov_report_e.longeval_gov_report_e_gen_54c5b0 import longeval_gov_report_e_datasets
    from .longeval_lines.longeval_lines_gen_9914f5 import longeval_lines_datasets
    from .longeval_passage_retrieval_zh.longeval_passage_retrieval_zh_gen_01cca2 import longeval_passage_retrieval_zh_datasets
    from .longeval_stackselect.longeval_stackselect_gen_03f1f8 import longeval_stackselect_datasets
    from .longeval_textsort.longeval_textsort_gen_28f0f7 import longeval_textsort_datasets
    from .longeval_trec_e.longeval_trec_e_gen_824187 import longeval_trec_e_datasets

longeval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
