from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3
from opencompass.partitioners import SizePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask

import torch

# longbench evaluation task
with read_base():
    from .datasets.longeval.longeval_2k.longeval import longeval_datasets

datasets = [*longeval_datasets]
