from mmengine.config import read_base

with read_base():
    from ..subjective_qa.subjectiveqav3_gen import subjectiveqav3_datasets
    from ..subjective_qa.alpaca_eval import alpaca_eval_datasets
    from ..subjective_qa.creation_bench import creation_bench_datasets
    from ..subjective_qa.ifeval import ifeval_datasets
    from ..subjective_qa.infobench import infobench_datasets
    from ..subjective_qa.labinfo import labinfo_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
