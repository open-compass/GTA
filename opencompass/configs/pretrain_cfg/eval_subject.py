from mmengine.config import read_base

from opencompass.models import OpenAI
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner, SlurmSequentialRunner
from opencompass.summarizers import PretrainSummarizer
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    # from .collections.example import datasets
    from .collections.subjectivity_only import datasets
    # from .models.huggingface import models
    # from .models.llama import models
    from .models.model_sub import models


work_dir = './outputs/subject/11_9/'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=3000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        # type=SlurmRunner,
        type=SlurmSequentialRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
        partition="llm_t",
        retry=3
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        retry=3,
        max_num_workers=2,
        partition="llm_t",
        task=dict(type=SubjectiveEvalTask,
                  judge_cfg=dict(abbr='GPT4.0',
                                 type=OpenAI, path='gpt-4',
                                 key="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",
                                 meta_template=api_meta_template,
                                 query_per_second=16,
                                 max_out_len=2048, max_seq_len=2048, batch_size=8)
                  )
    ),

)


# python run.py configs/eval_subject.py -p llm_t -r -l --debug 2>&1 | tee log.txt


# python run.py configs/pretrain/eval_subject.py -p llm -r -l --debug 2>&1 | tee log.txt
