from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask
from opencompass.models import OpenAI


with read_base():
    # choose a list of datasets
    from .datasets.collections.subjectivity_only import datasets
    # choose a model of interest
    from .models.hf_internlm.hf_internlm_7b import models
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLEvalTask,
                  judge_cfg=dict(abbr='GPT3.5',
                                 type=OpenAI, path='gpt-3.5-turbo',
                                 key="xxxx",
                                 meta_template=api_meta_template,
                                 query_per_second=1,
                                 max_out_len=2048, max_seq_len=2048, batch_size=2)
                  )),
)
