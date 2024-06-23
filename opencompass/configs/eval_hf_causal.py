from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    from .summarizers.medium import summarizer
    from .lark import lark_bot_url
    from .datasets.collections.base_small import datasets

# TODO: Del proxy related config before opensourcing
#  hf_proxies = None
hf_proxies = {
    'http': 'http://10.1.8.5:32680',
    'https': 'http://10.1.8.5:32680',
}

models = [
    # OPT-2.7b
    dict(
       type=HuggingFaceCausalLM,
       path='facebook/opt-2.7b',
       tokenizer_path='facebook/opt-2.7b',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=hf_proxies,
           trust_remote_code=True,
       ),
       max_out_len=100,
       max_seq_len=2048,
       batch_size=16,
       model_kwargs=dict(device_map='auto', revision='397f71a473a150c00f0fe3fc4a2f78ff3ccaf82d'),
       run_cfg=dict(num_gpus=1, num_procs=1),
    )

    # Baichuan-7B
    # dict(
    #    type=HuggingFaceCausalLM,
    #    path='baichuan-inc/baichuan-7B',
    #    tokenizer_path='baichuan-inc/baichuan-7B',
    #    tokenizer_kwargs=dict(
    #        padding_side='left',
    #        truncation_side='left',
    #        proxies=hf_proxies,
    #        trust_remote_code=True,
    #    ),
    #    max_out_len=256,
    #    max_seq_len=2048,
    #    batch_size=1,
    #    extract_pred_after_decode=True,
    #    model_kwargs=dict(device_map='auto',
    #                      trust_remote_code=True,
    #                      revision='39916f64eb892ccdc1982b0eef845b3b8fd43f6b'),
    #    run_cfg=dict(num_gpus=1, num_procs=1),
    # )

    # LLaMA 65B
    # dict(
    #     type=HuggingFaceCausalLM,
    #     path='huggyllama/llama-65b',
    #     tokenizer_path='huggyllama/llama-65b',
    #     tokenizer_kwargs=dict(
    #         padding_side='left',
    #         truncation_side='left',
    #         proxies=hf_proxies,
    #         trust_remote_code=True,
    #     ),
    #     max_out_len=25,
    #     max_seq_len=2048,
    #     batch_size=1,
    #     extract_pred_after_decode=True,
    #     model_kwargs=dict(device_map='auto', revision='49707c5313d34d1c5a846e29cf2a2a650c22c8ee'),
    #     run_cfg=dict(num_gpus=8, num_procs=1),
    # )

    # Falcon-40B
    # dict(
    #     type=HuggingFaceCausalLM,
    #     path='tiiuae/falcon-40b-instruct',
    #     tokenizer_path='tiiuae/falcon-40b-instruct',
    #     tokenizer_kwargs=dict(
    #         padding_side='left',
    #         truncation_side='left',
    #         proxies=hf_proxies,
    #         trust_remote_code=True,
    #     ),
    #     max_out_len=25,
    #     max_seq_len=2048,
    #     batch_size=4,
    #     model_kwargs=dict(trust_remote_code=True, device_map='auto', revision='5b9409410d251ab8e06c48078721c8e2b71fa8a1'),
    #     run_cfg=dict(num_gpus=8, num_procs=1),
    # )
]
