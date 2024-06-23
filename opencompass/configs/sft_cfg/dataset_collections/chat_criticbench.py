from mmengine.config import read_base

with read_base():
    from ...datasets.criticbench.criticbench_gen import criticbench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
