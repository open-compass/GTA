from mmengine.config import read_base

with read_base():
    from .longeval_summarization_en_gen_f39608 import longeval_summarization_en_datasets  # noqa: F401, F403
