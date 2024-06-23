from mmengine.config import read_base
from opencompass.models.internal import InternLMwithModule
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner, SlurmSequentialRunner
from opencompass.summarizers import SubjectiveSummarizer, PretrainSummarizer
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from copy import deepcopy
import os.path as osp

with read_base():
    from .collections.chat_medium2 import datasets

    from .models.model_11_6.models_p import models
    from .summarizers.medium_report import summarizer

# datasets 在 from ..datasets.collections.chat_medium import datasets 已经设置好了
# datasets = [..]
work_dir = './outputs/2023_11_6/'


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=32,
        partition="llm_o",
        task=dict(type=OpenICLInferTask),
        # retry=3
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner,n=10),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=64,
        partition="llm_t",
        task=dict(type=OpenICLEvalTask,dump_details=True),
        # retry=4
    ),
)

meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<TOKENS_UNUSED_1>\n', generate=True),
    ],
    eos_token_id=103028
)


#  python run.py configs/pretrain/eval_chat.py -p llm_t -r --debug
