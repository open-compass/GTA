from opencompass.partitioners import SizePartitioner, NaivePartitioner, InferTimePartitioner
from opencompass.runners import DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
import os

workspace_id = 'wsf0b8joz85cuz0p'
worker_image = 'pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/chenxun-st:llm-test'

# users can also use the old version aliyun_cfg
# NOTE: `bashrc_path` has higher priority than `python_env_path`
# aliyun_cfg = dict(
#     bashrc_path=f"/cpfs01/user/{os.getenv('USER')}/.bashrc",
#     conda_env_name='opencompass_sft_internal',
#     dlc_config_path=f"{os.getenv('HOME')}/.dlc/config",
#     workspace_id=workspace_id,
#     worker_image=worker_image
# )

aliyun_cfg = dict(
    # users can also use own conda env
    # e.g. python_env_path='/cpfs01/user/wangyudong/miniconda3/envs/opencompass_internal',
    python_env_path='/cpfs01/shared/public/yehaochen/conda_env/shared_opencompass',
    dlc_config_path=f"{os.getenv('HOME')}/.dlc/config",
    workspace_id=workspace_id,
    worker_image=worker_image,
    hf_offline=True,

    # optional, suggest to set the http_proxy if `hf_offline` if False.
    # http_proxy='http://58.34.83.134:31128/',

    # optional, using mirror to speed up the huggingface download
    # hf_endpoint='https://hf-mirror.com',

    # optional, if not set, will use the default cache path
    huggingface_cache='/cpfs01/shared/public/public_hdd/wangyudong/huggingface_cache/huggingface/hub',
    torch_cache='/cpfs01/shared/public/public_hdd/wangyudong/torch_cache'
)

# new infer setting, using split strategy, may speed up the infer time
infer = dict(
    partitioner=dict(
        type=InferTimePartitioner,
        max_task_time=3600,
        strategy='split'),
    runner=dict(
        type=DLCRunner,
        max_num_workers=64,
        retry=4,
        aliyun_cfg=aliyun_cfg,
        task=dict(type=OpenICLInferTask)
    ),
)

# origin infer setting
infer_size = dict(
    partitioner=dict(
        type=SizePartitioner, 
        max_task_size=5000,     # default = 2000
        gen_task_coef=20,
    ),
    runner=dict(
        type=DLCRunner,
        max_num_workers=64,
        retry=4,
        aliyun_cfg=aliyun_cfg,
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=aliyun_cfg,
        max_num_workers=128,
        retry=2,
        task=dict(type=OpenICLEvalTask)),
)
