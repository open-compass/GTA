from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.summarizers import SubjectiveSummarizer, PretrainSummarizer
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .collections.C_plus import datasets
    # from .collections.example import datasets
    from .collections.base_small import datasets
    # from .collections.base_small_pre import datasets

    # from .models.llama import models
    from .models.model_10_29.models import models
    # from .models.huggingface import models

    from .summarizers.small import summarizer



work_dir = './outputs/2023_10_29/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=100,
        task=dict(type=OpenICLInferTask),
        retry=4),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask,dump_details=True),
        retry=3),
)
summarizer = dict(
    type=PretrainSummarizer
)

# python run.py configs/pretrain/eval_s.py -p llm  -q spot -r -l --debug 2>&1 | tee log.txt
