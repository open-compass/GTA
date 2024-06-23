from mmengine.config import read_base

with read_base():
    from ..subjective_qa.chat_enhance import subjective_chat_enhance_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
