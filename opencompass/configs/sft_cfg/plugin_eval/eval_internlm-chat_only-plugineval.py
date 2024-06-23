from mmengine.config import read_base
from opencompass.openicl import ChatInferencer

with read_base():
    from ...datasets.pluginEval.plugin_eval_v2_gen import (
        plugin_eval_datasets as datasets
    )
    from ..summarizers.medium_chat_sft_v051_plugineval_v2 import summarizer
    from ..lark import lark_bot_url

    from ...internal.models.regression_20231031_S.InternLM_7B_Chat_v1_1 import models as InternLM_7B_Chat_model
    from ...internal.models.regression_20231031_S.InternLM_20B_Chat import models as InternLM_20B_Chat_model

    from ...internal.clusters.slurm_S import infer, eval

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='SYSTEM', begin='<|System|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

infer["runner"]["partition"] = 'llmit'
eval["runner"]["partition"] = 'llmit'
infer["runner"]["quotatype"] = 'reserved'
eval["runner"]["quotatype"] = 'reserved'

for d in datasets:
    d['infer_cfg']['inferencer'] = dict(type=ChatInferencer)
    d["infer_cfg"]["inferencer"]["save_every"] = 1

models = sum([v for k, v in locals().items() if k.endswith("_model")], [])
for model in models:
    model["model_kwargs"]=dict(
        trust_remote_code=True,
        device_map='auto',
        # cache_dir='~/.cache/huggingface/hub',
    )
    model["tokenizer_kwargs"]=dict(
        padding_side='left',
        truncation_side='left',
        use_fast=False,
        trust_remote_code=True,
        # cache_dir='~/.cache/huggingface/hub',
    )
    model["meta_template"] = _meta_template
    model["max_seq_len"] = 8192
    model["batch_size"] = 4
    model["run_cfg"] = dict(num_gpus=1, num_procs=1)

summarizer['dataset_abbrs'] = [i for i in summarizer['dataset_abbrs'] if isinstance(i, (str, list))]
