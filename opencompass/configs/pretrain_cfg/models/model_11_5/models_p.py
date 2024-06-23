meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<TOKENS_UNUSED_1>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    # dict(
    #     abbr='oppenheimer_7B_2.1.1_2000',
    #     type='opencompass.models.internal.InternLMwithModule',
    #     path='s3://checkpoints_ssd_02/oppenheimer/oppenheimer_7B_2.1.1/2000',
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path='/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.2.0dev_10_16',
    #     model_config='/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.2.0dev_10_16/configs/oppenheimer_7B_2.1.1.py',
    #     model_type='LLAMA',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr='oppenheimer_7B_2.1.1_4000',
    #     type='opencompass.models.internal.InternLMwithModule',
    #     path='s3://checkpoints_ssd_02/oppenheimer/oppenheimer_7B_2.1.1/4000',
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path='/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.2.0dev_10_16',
    #     model_config='/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.2.0dev_10_16/configs/oppenheimer_7B_2.1.1.py',
    #     model_type='LLAMA',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),
    # dict(
    #     abbr='oppenheimer_7B_2.1.1_6000',
    #     type='opencompass.models.internal.InternLMwithModule',
    #     path='s3://checkpoints_ssd_02/oppenheimer/oppenheimer_7B_2.1.1/6000',
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    #     tokenizer_type='llama',
    #     module_path='/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.2.0dev_10_16',
    #     model_config='/mnt/petrelfs/feizhaoye.dispatch/train_internlm_v0.2.0dev_10_16/configs/oppenheimer_7B_2.1.1.py',
    #     model_type='LLAMA',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    # dict(
    #     abbr='7B_sft_wm_v0.9',
    #     type='opencompass.models.internal.InternLMwithModule',
    #     path='/mnt/inspurfs/gaoyang/models//7B_sft_wm_v0.9/5260',
    #     tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    #     tokenizer_type='v7',
    #     module_path='/mnt/petrelfs/share_data/gaoyang/InternLM_train_13b',
    #     model_config='/mnt/petrelfs/llmit/code/train_7b/InternLM/RUN/maibao_kaoshi_7_5_ST_8k_v0213rc8/09-05-19:57:40/7b_maibao_8k.py',
    #     model_type='INTERNLM',
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=1,
    #     meta_template=meta_template,
    #     run_cfg=dict(num_gpus=1, num_procs=1)
    # ),

    dict(
        abbr='20B_sft_wm_v0.2.4',
        type='opencompass.models.internal.InternLMwithModule',
        path='/mnt/inspurfs/gaoyang/models/20B_sft_wm_v0.1_codebase-v0.2.4/2795',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
        tokenizer_type='v7',
        module_path='/mnt/petrelfs/share_data/gaoyang/train_internlm_v0.2.4/',
        model_config='/mnt/petrelfs/share_data/gaoyang/train_internlm_v0.2.4/RUN/20B_sft_wm_v0.1_codebase-v0.2.4/11-03-14:31:11/20B_sft_wm_v0.1_codebase-v0.2.4.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=4, num_procs=4)
    ),
]
