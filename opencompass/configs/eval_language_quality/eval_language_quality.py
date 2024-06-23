from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner, InferTimePartitioner
from opencompass.runners import LocalRunner, SlurmRunner, SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.models import OpenAI


with read_base():
    from ..datasets.language_quality.expression import expression_base
    from ..datasets.language_quality.expression import expression_api
    from ..datasets.language_quality.commonsense import commonsense_base
    from ..datasets.language_quality.commonsense import commonsense_api
    from ..datasets.language_quality.logic import logic_base
    from ..datasets.language_quality.logic import logic_api
    from .models import base_models, api_models 
model_dataset_combinations = []
work_dir = './outputs/2023_12_11_v2/'

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
    # partitioner=dict(type=SizePartitioner, max_task_size=30000, gen_task_coef=5),
    #partitioner=dict(type=InferTimePartitioner, max_task_time=5000),
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=SlurmSequentialRunner,
        partition="llm2_t",
        retry=0,
        max_num_workers=9,
        task=dict(type=OpenICLInferTask),
        ),
)

# if you don't have an openai key or don't want to use API-based scores, please uncomments the following ICLEval and comments the SubjectiveEval.
# A few evalulation will be broken but others can work without API-based models. 

# eval = dict(
#     partitioner=dict(type=NaivePartitioner,n=1),
#     runner=dict(
#         type=SlurmSequentialRunner,
#         partition="llm2_t",
#         max_num_workers=1,
#         task=dict(type=OpenICLEvalTask, dump_details=True),
#         retry=0),
# )

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        retry=0,
        max_num_workers=4,
        partition="llm2_t",
        task=dict(type=SubjectiveEvalTask,
                  judge_cfg=dict(abbr='GPT3.5',
                                 type=OpenAI, path='gpt-3.5-turbo-1106',
                                 key="fill your openai API key",
                                 meta_template=api_meta_template,
                                 query_per_second=1,
                                 max_out_len=2048, max_seq_len=2048, batch_size=4)
                  )
    ),
)

models = base_models[:1] # taking one model for testing
datasets = expression_base + logic_base + commonsense_base
for d in datasets:
    d["infer_cfg"]["inferencer"]["save_every"] = 1

model_dataset_combinations.append(dict(models=base_models[:1], datasets=expression_base + logic_base + commonsense_base))
# beforing running this file, please ensure a backend server is running and the address is assigned to the enviroment called OPNECOMPASS_LQ_BACKEND
# For example, export OPENCOMPASS_LQ_BACKEND = 'http://127.0.0.1:5001/'
# python -u run.py configs/eval_language_quality/eval_language_quality.py -p llm2_t -s -r 
