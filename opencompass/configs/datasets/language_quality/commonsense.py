from mmengine.config import read_base

with read_base():
    from .commonsense.commonsenseqa_gen import commonsenseqa_gen_datasets
    from .commonsense.commonsenseqa_ppl import commonsenseqa_ppl_datasets
    from .commonsense.conceptnet_llm import conceptnet_datasets
    from .commonsense.instruction_llm import instruction_datasets
    from .commonsense.k12_llm import k12_datasets
    from .commonsense.story_gen import story_gen_datasets
    from .commonsense.story_ppl import story_ppl_datasets

shared = conceptnet_datasets + instruction_datasets + k12_datasets
commonsense_base = shared + commonsenseqa_ppl_datasets + story_ppl_datasets
commonsense_api = shared + commonsenseqa_gen_datasets + story_gen_datasets
