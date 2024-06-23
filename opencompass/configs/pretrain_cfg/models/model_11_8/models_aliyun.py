models = [
    # dict(
    #     abbr="official_qianxuesen_7B_v1_0_3_253500",
    #     type="opencompass.models.internal.InternLMwithModule",
    #     path="/cpfs01/shared/alillm2/alillm2_hdd/lvhaijun/ckpts/official_qianxuesen_7B_v1.0.1/253500",
    #     tokenizer_path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/tokenizers/llama.model",
    #     tokenizer_type="llama",
    #     module_path="/cpfs01/shared/public/lvhaijun/train_internlm_attribute_v2",
    #     model_config="/cpfs01/shared/public/lvhaijun/train_internlm_attribute_v2/configs/official_qianxuesen_7B_v1.0.3.py",
    #     model_type="LLAMA",
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=4,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
]

models += [
    dict(                
        abbr="official_qianxuesen_base_500_7B_v1.0.1_60000",                
        type="opencompass.models.internal.InternLMwithModule",                
        path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/ckpts/official_qianxuesen_base_500_7B_v1.0.1/60000",
        tokenizer_path="/cpfs01/shared/alillm2/alillm2_hdd/zhangshuo/tokenizers/llama.model",                
        tokenizer_type="llama",                
        module_path="/cpfs01/shared/public/zhangshuo/train_internlm/",                
        model_config="/cpfs01/shared/public/zhangshuo/train_internlm/configs/official_qianxuesen_base_500_7B_v1.0.1.py",                
        model_type="LLAMA",                
        max_out_len=100,                
        max_seq_len=2048,                
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),                
    ),
]