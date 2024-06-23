from mmengine.config import read_base

with read_base():
    from .compassbench_v1_math_ppl_e71e76 import compassbench_v1_math_datasets  # noqa: F401, F40
