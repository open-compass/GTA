from mmengine.config import read_base
from opencompass.models.internal import OpenAIAssistantAllesAPIN
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets as datasets

models = [
    dict(abbr='GPT4-assistant',
        type=OpenAIAssistantAllesAPIN,
        path='gpt-4',
        url='https://openxlab.org.cn/gw/alles-apin-hub/v1/openai/v1',
        key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=1),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)
