from mmengine.config import read_base

with read_base():
    # from ..datasets.tinystories.tinystories_gen import tinystories_datasets
    # from ..datasets.subjectivity.subjectivity import subjectivity_datasets
    # from ..datasets.subjectivity.story import story_datasets

    # from ...datasets.subjectivity.fallacy_attack import fallacy_attack_datasets
    # from ...datasets.subjectivity.contradiction import contradiction_datasets
    # from ...datasets.subjectivity.cot import cot_datasets
    # from ...datasets.subjectivity.icl import icl_datasets
    # from ...datasets.subjectivity.coreference_resolution import coreference_resolution_datasets
    from ...datasets.chinese_language_base.chinese_language_base_abc import chinese_language_base_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
