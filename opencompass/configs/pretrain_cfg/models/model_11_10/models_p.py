models = [
]

models += [
    dict(
        abbr="good_7b_200000",
        type="opencompass.models.internal.InternLMwithModule",
        path="/mnt/inspurfs/share_data/llm_data/yangxiaogui/checkpoints/good_7b/200000",
        tokenizer_path="/mnt/petrelfs/share_data/yanhang/tokenizes/v11.model",
        tokenizer_type="llama",
        module_path="/mnt/petrelfs/share_data/yangxiaogui/train_internlm_good_7b",
        model_config="/mnt/petrelfs/share_data/yangxiaogui/train_internlm_good_7b/configs/good_7b.py",
        model_type="BAICHUAN2",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
