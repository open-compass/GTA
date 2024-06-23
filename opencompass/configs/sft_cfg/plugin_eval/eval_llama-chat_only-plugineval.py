from mmengine.config import read_base
from opencompass.openicl import ChatInferencer

with read_base():
    from ...datasets.pluginEval.plugin_eval_v2_gen import (
        plugin_eval_datasets as datasets
    )
    from ..summarizers.medium_chat_sft_v051_plugineval_v2 import summarizer
    from ..lark import lark_bot_url

    from ...models.hf_llama.hf_llama2_7b_chat import models as llama2_7b_chat_model
    from ...models.hf_llama.hf_llama2_13b_chat import models as llama2_13b_chat_model
    from ...models.hf_llama.hf_llama2_70b_chat import models as llama2_70b_chat_model

    from ...internal.clusters.slurm_S import infer, eval

_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN', begin='[INST]', end='[\INST]'),
        dict(role='SYSTEM', api_role='HUMAN', begin='[INST]', end='[\INST]'),
        dict(role='BOT',
                api_role='BOT',
                begin='',
                end='</s>',
                generate=True),
    ],
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
        cache_dir='/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/'
    )
    model["tokenizer_kwargs"]=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        cache_dir='/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/'
    )
    model["max_seq_len"] = 8192
    model["batch_size"] = 8
    model["meta_template"] = _meta_template
    model["force_bf16"] = True
    # model["skip_overlength"] = True  # skip to infer and set a null return when the sequence length is greater than the maximum length

summarizer['dataset_abbrs'] = [i for i in summarizer['dataset_abbrs'] if isinstance(i, (str, list))]
