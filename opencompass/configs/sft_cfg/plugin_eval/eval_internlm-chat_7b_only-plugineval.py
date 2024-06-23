from mmengine.config import read_base
from opencompass.openicl import ChatInferencer
from opencompass.models.internal import InternLMwithModule
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from copy import deepcopy
import os.path as osp

with read_base():
    from ...datasets.pluginEval.plugin_eval_v2_gen import (
        plugin_eval_datasets as datasets
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
        dict(role='HUMAN', begin='<TOKENS_UNUSED_140>user\n', end='<TOKENS_UNUSED_139>\n'),
        dict(role='SYSTEM', begin='<TOKENS_UNUSED_140>system\n', end='<TOKENS_UNUSED_139>\n'),
        dict(role='BOT', begin='<TOKENS_UNUSED_140>assistant\n', end='<TOKENS_UNUSED_139>\n', generate=True),
    ],
    eos_token_id=103166)

# _without_meta_template = dict(
#     begin="""""",
#     round=[
#         dict(role='HUMAN', begin='<|User|>:', end='\n'),
#         dict(role='SYSTEM', begin='<|System|>:', end='\n'),
#         dict(role='BOT', begin='<|Bot|>:', end='<TOKENS_UNUSED_1>\n', generate=True),
#     ],
#     eos_token_id=103028)

for d in datasets:
    d['infer_cfg']['inferencer'] = dict(type=ChatInferencer)

base_dict = dict(
    abbr=None,
    path=None,
    type=InternLMwithModule,
    model_type='INTERNLM',
    tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    tokenizer_type='v7',
    module_path="/mnt/petrelfs/chenzehui/code/train_internlm",
    model_config="/mnt/petrelfs/chenzehui/code/train_internlm/configs/internlm_7b_8k_sft.py",
    meta_template=_without_meta_template,
    max_out_len=100,
    # If want to use the full length of the model, set max_seq_len=8192, otherwise can set max_seq_len=2048.
    max_seq_len=8192,
    batch_size=4,
    # using bf16 may decrease the performance, force set to fp16
    model_dtype='torch.float16',
    run_cfg=dict(
        num_gpus=1,
        num_procs=1))

models_path = [
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231212/toolbench_baseline_json/560",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231212/toolbench_baseline_sq/560",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231213/toolbench_tflan_baseline/680",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231213/toolbench_tflanv1_czh/2040",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231213/toolbench_tflan_baseline_instruct/680",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231214/toolbench_tflan09_react01/670",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231214/toolbench_tflan75_react25/650",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231214/toolbench_tflan_baseline_instruct10000/690",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231214/toolbench_tflan9_react1_instruct5000/680",

    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231216/toolbench_t9r1s1_ins5k/370",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231217/tb_ms_ronly/360",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231217/tb_ms_t9r1_ins5k/620",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231217/tb_ms_t9r1s1_ins5k/660",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231217/tb_ms_t9r1s1c1_ins5k/700",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231218/tb_ms_t9r1s1c1_ins_j4s2/670",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231218/tb_ms_t9r1s2c1_inst_j4s2/720",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231218/tb_ms_t9r1s1c1_ins_j4s2_05/340",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231219/tb_ms_t9r1s2c1_inst_j8s4_05/360",
    # "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231219/tb_ms_t9r1s2c1_inst_j8s4_mix_gor_rewoo_05/370",
    "p_llmit:s3://llmit/sft/ckpt/toolbench_0830/20231221/tb_ms_t9r1s2c1_inst_j4s2_v2_05/270"
]

models = []
for model_path in models_path:
    tmp_model_dict = deepcopy(base_dict)
    if model_path.endswith('/'):
        model_path = model_path[:-1]
    # abbr = osp.split(osp.split(model_path)[0])[-1]
    abbr = model_path.split("/")[-2]
    tmp_model_dict['abbr'] = abbr
    tmp_model_dict['path'] = model_path
    models.append(tmp_model_dict)

del models_path, model_path, tmp_model_dict, abbr, base_dict
