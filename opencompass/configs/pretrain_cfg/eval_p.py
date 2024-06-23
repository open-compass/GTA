from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner, SlurmSequentialRunner
from opencompass.summarizers import SubjectiveSummarizer, PretrainSummarizer
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .collections.C_plus import datasets
    # from .collections.example import datasets
    from .collections.base_small import datasets
    # from .collections.base_small_pre import datasets

    # from .models.llama import models
    # from .models.huggingface import models
    from .models.model_11_10.models_p import models

    from .summarizers.pretrain import summarizer
    # from .summarizers.leizhikuai import summarizer


work_dir = './outputs/2023_11_10/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=500, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmSequentialRunner,
        # type=SlurmRunner,
        # quotatype='reserved',
        partition="llm_o",
        retry=3,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner,n=10),
    runner=dict(
        type=SlurmSequentialRunner,
        partition="llm_t",
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask,dump_details=True),
        retry=4),
)


# python run.py configs/eval_p.py -p p4_test -r -l --debug 2>&1 | tee log.txt

# python run.py configs/pretrain/eval_p.py -p llm_t -r -l --debug 2>&1 | tee log.txt
