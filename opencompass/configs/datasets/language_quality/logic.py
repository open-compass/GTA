from mmengine.config import read_base

with read_base():
    from .logic.contradiction import contradiction_datasets
    from .logic.coreference_resolution import coreference_resolution_datasets
    from .logic.cot import cot_datasets
    from .logic.fallacy_attack import fallacy_attack_datasets
    from .logic.icl_cn import icl_cn_datasets
    from .logic.icl_en import icl_en_datasets
    from .logic.winograd_cn_api import winograd_cn_api_datasets
    from .logic.winograd_cn_ppl import winograd_cn_ppl_datasets
    from .logic.winograd_en_api import winograd_en_api_datasets
    from .logic.winograd_en_ppl import winograd_en_ppl_datasets

#contradiction_datasets = []
shared = contradiction_datasets + coreference_resolution_datasets + cot_datasets + fallacy_attack_datasets + \
    icl_cn_datasets + icl_en_datasets

logic_base = shared + winograd_cn_ppl_datasets + winograd_en_ppl_datasets
logic_api = shared + winograd_cn_api_datasets + winograd_en_api_datasets
