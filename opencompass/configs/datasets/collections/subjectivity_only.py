from mmengine.config import read_base

with read_base():
    from ..subjectivity.subjectivity import subjectivity_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
