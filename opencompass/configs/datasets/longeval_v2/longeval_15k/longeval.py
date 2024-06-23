from mmengine.config import read_base

with read_base():
    from .longeval_classification_en.longeval_classification_en_gen import longeval_classification_en_datasets
    from .longeval_lines.longeval_lines_gen import longeval_lines_datasets
    from .longeval_qa_en.longeval_qa_en_gen import longeval_qa_en_datasets
    from .longeval_qa_zh.longeval_qa_zh_gen import longeval_qa_zh_datasets
    from .longeval_stackselect.longeval_stackselect_gen import longeval_stackselect_datasets
    from .longeval_summarization_en.longeval_summarization_en_gen import longeval_summarization_en_datasets
    from .longeval_textsort.longeval_textsort_gen import longeval_textsort_datasets

longeval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
