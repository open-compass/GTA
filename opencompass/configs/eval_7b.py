from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    from .summarizers.small import summarizer
    from .lark import lark_bot_url
    from .datasets.collections.base_small import datasets

models = [
    # dict(abbr='InternLM-7B-5177@77999',
    #     type='LLMv2', path="opennlplab_hdd_new:s3://shared_weight/huggingface/pretraining/internlm-7b/internlm-7b-5177_pj_v4_labelsm/77999",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model', tokenizer_type='v7', model_type="origin",
    #     max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1)),
    dict(abbr='llama-7b-hf',
        type=HuggingFaceCausalLM, path="decapoda-research/llama-7b-hf", tokenizer_path='decapoda-research/llama-7b-hf',
        tokenizer_kwargs=dict(padding_side='left', use_fast=False, proxies={'http': 'http://10.1.8.5:32680', 'https': 'http://10.1.8.5:32680'}),
        max_out_len=100, max_seq_len=2048, batch_size=8, model_kwargs=dict(device_map='auto'), run_cfg=dict(num_gpus=2, num_procs=1)),
]
