from mmengine.config import read_base

with read_base():
    from .longeval_qa_en_gen_c20042 import longeval_qa_en_datasets  # noqa: F401, F403
