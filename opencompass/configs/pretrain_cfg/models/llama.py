from opencompass.models import OpenAI, HuggingFaceCausalLM
from opencompass.models.internal import LLama, InternLMwithModule

api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    # dict(abbr="LLama7B",
    #      type=LLama, path='/mnt/petrelfs/share_data/llm_llama/7B',
    #      tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model', tokenizer_type='llama',
    #      max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1)),
    # dict(abbr="LLama13B",
    #      type=LLama, path='/mnt/petrelfs/share_data/llm_llama/13B',
    #      tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model', tokenizer_type='llama',
    #      max_out_len=100, max_seq_len=2048, batch_size=1, run_cfg=dict(num_gpus=2, num_procs=2)),
    #
    dict(abbr="LLama2-7B",
         type=LLama, path='/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-7b',
         tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model', tokenizer_type='llama',
         max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1)),
    # dict(abbr="LLama2-13B",
    #      type=LLama, path='/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-13b',
    #      tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model', tokenizer_type='llama',
    #      max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=2, num_procs=2)),
    #
    # dict(abbr='GPT4',
    #      type=OpenAI, path='gpt-4-0613',
    #      key='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',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    #      meta_template=api_meta_template,
    #      query_per_second=1,
    #      max_out_len=2048, max_seq_len=2048, batch_size=8),
    # dict(
    #     abbr="qwen2-7b_0",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/share_data/feizhaoye/huggingface/Qwen/Qwen-7B_9_25/",
    #     # tokenizer_path='chiayewken/Qwen-7B',
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False,
    #                           # revision='39fc5fdcb95c8c367bbdb3bfc0db71d96266de09'
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True,
    #                       #
    #                       # revision='39fc5fdcb95c8c367bbdb3bfc0db71d96266de09'
    #                       ),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     pad_token_id=0,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    # dict(
    #     abbr="chatglm3",
    #     type=HuggingFaceCausalLM,
    #     path="/mnt/petrelfs/share_data/feizhaoye/huggingface/chatglm3/",
    #     # tokenizer_path='Qwen/Qwen-7B',
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           trust_remote_code=True,
    #                           use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True,
    #                       ),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    # dict(
    #     abbr="baichuan2_7b_base_02420",
    #     type=InternLMwithModule,
    #     model_type="BAICHUAN2",
    #     path="/mnt/petrelfs/share_data/common_share/baichuan2_7b_base_intermediate_checkpoints/train_02420B/",
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/baichuan2.model',
    #     tokenizer_type='llama',
    #     module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/",
    #     model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm/configs/_base_/models/baichuan2/baichuan2_7B.py",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
]
