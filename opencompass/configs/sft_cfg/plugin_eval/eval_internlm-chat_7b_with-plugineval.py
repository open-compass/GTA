from mmengine.config import read_base
from opencompass.openicl import ChatInferencer
from opencompass.models.internal import InternLMwithModule
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from copy import deepcopy
import os.path as osp

with read_base():
    from ..dataset_collections.medium_chat_sft_v051_plugineval import (
        datasets
    )
    from ..summarizers.medium_chat_sft_v051_plugineval_v2 import summarizer
    from ..lark import lark_bot_url

# datasets 在 from ..datasets.collections.chat_medium import datasets 已经设置好了
# datasets = [...]

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=5000,     # default = 2000
        gen_task_coef=20,
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=128,
        retry=4,
        partition='llmit',
        quotatype='reserved',
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmit',
        quotatype='reserved',
        max_num_workers=128,
        retry=2,
        task=dict(type=OpenICLEvalTask)),
)

_without_meta_template = dict(
    begin="""""",
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='SYSTEM', begin='<|System|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<TOKENS_UNUSED_1>\n', generate=True),
    ],
    eos_token_id=103028)

for d in datasets:
    d['infer_cfg']['inferencer'] = dict(type=ChatInferencer)

base_dict = dict(
    abbr=None,
    path=None,
    type=InternLMwithModule,
    model_type='INTERNLM',
    tokenizer_path='/mnt/petrelfs/llmit/tokenizers/V7.model',
    tokenizer_type='v7',
    module_path="/mnt/petrelfs/llmit/code/opencompass_internal/sft_opencompass_v052/train_internlm",
    model_config="/mnt/petrelfs/llmit/code/opencompass_internal/sft_opencompass_v052/train_internlm/configs/maibao_7b_8k_sft.py",
    meta_template=_without_meta_template,
    max_out_len=100,
    # If want to use the full length of the model, set max_seq_len=8192, otherwise can set max_seq_len=2048.
    max_seq_len=8192,
    batch_size=8,
    # using bf16 may decrease the performance, force set to fp16
    model_dtype='torch.float16',
    run_cfg=dict(
        num_gpus=1,
        num_procs=1))

models_path = [
    '/mnt/petrelfs/liujiangning/work/dev/ftdp/data/models/msagent/plugin_v02_tool_d1116rc2/llamav7_8k/n2_m8_ir3v4/sft_7b_t_msagent/510',
]

models = []

for model_path in models_path:
    tmp_model_dict = deepcopy(base_dict)
    if model_path.endswith('/'):
        model_path = model_path[:-1]
    # abbr = osp.split(osp.split(model_path)[0])[-1]
    abbr = model_path.split("/")[-5]
    tmp_model_dict['abbr'] = abbr
    tmp_model_dict['path'] = model_path
    models.append(tmp_model_dict)

del models_path, model_path, tmp_model_dict, abbr, base_dict
