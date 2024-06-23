from mmengine.config import read_base
from opencompass.models.internal import PJLM
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from ..summarizers.medium import summarizer
    from ..datasets.ceval.ceval_gen import ceval_datasets

datasets = [
    *ceval_datasets,
]

# Make sure you have the `openxlab` installed, and have
# run `openxlab config` to set your username and password.
models = [
    dict(
        abbr='PJLM-chat',
        type=PJLM,
        path='xxxxxxxxx',
        key=None, # Can be None and will be set automatically.
        url='xxxxxxxxx',
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=2,
        concurrent_users=2,
        task=dict(type=OpenICLInferTask)),
)

work_dir = "outputs/api_pjlm/"
