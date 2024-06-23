from mmengine.config import read_base
from opencompass.openicl import ChatInferencer

with read_base():
    from ...datasets.pluginEval.plugin_eval_v2_gen import (
        plugin_eval_datasets as datasets
    )
    from ..summarizers.medium_chat_sft_v051_plugineval_v2 import summarizer
    from ..lark import lark_bot_url

    from ...models.qwen.hf_qwen_7b_chat import models as qwen_7b_chat_model
    from ...models.qwen.hf_qwen_14b_chat import models as qwen_14b_chat_model
    from ...models.qwen.hf_qwen_72b_chat import models as qwen_72b_chat_model


    from ...internal.clusters.slurm_S import infer, eval

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='SYSTEM', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='BOT',
                begin='\n<|im_start|>assistant\n',
                end='<|im_end|>',
                generate=True),
    ]
)

infer["runner"]["partition"] = 'llmit'
eval["runner"]["partition"] = 'llmit'
infer["runner"]["quotatype"] = 'auto'
eval["runner"]["quotatype"] = 'auto'

for d in datasets:
    d['infer_cfg']['inferencer'] = dict(type=ChatInferencer)
    d["infer_cfg"]["inferencer"]["save_every"] = 1

models = sum([v for k, v in locals().items() if k.endswith("_model")], [])
for model in models:
    model["model_kwargs"]=dict(
        device_map='auto',
        trust_remote_code=True,
        # cache_dir='/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/'
    )
    model["tokenizer_kwargs"]=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        use_fast=False,
        # cache_dir='/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/'
    )
    model["max_seq_len"] = 8192
    model["batch_size"] = 4
    model["meta_template"] = _meta_template
    # model["run_cfg"] = dict(num_gpus=1, num_procs=1)

summarizer['dataset_abbrs'] = [i for i in summarizer['dataset_abbrs'] if isinstance(i, (str, list))]
