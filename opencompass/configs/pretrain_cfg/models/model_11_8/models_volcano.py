# from opencompass.models import HuggingFaceCausalLM, InternLM, LLMv4, LLMv3, LLama
from opencompass.models.internal import InternLM, LLMv_ling, LLMv_zhangshuo

models = [

dict(
        abbr='TARO_7B_v0_2_0_1000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_2_0/1000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_2_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_2_0_2000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_2_0/2000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_2_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_2_0_3000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_2_0/3000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_2_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),

dict(
        abbr='TARO_7B_v0_3_0_1000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_3_0/1000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_3_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_3_0_2000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_3_0/2000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_3_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_3_0_3000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_3_0/3000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_3_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_4_0_1000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_4_0/1000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_4_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_4_0_2000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_4_0/2000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_4_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),
dict(
        abbr='TARO_7B_v0_4_0_3000',
        type='opencompass.models.internal.InternLMwithModule',
        path='/fs-computility/llm/shared/llm_data/ckpts/leizhikai/TARO_7B_v0_4_0/3000',
        tokenizer_path='/fs-computility/llm/shared/llm_data/yanhang/tokenizers/llama.model',
        tokenizer_type='llama',
        module_path='/fs-computility/llm/shared/leizhikai/train_internlm',
        model_config='/fs-computility/llm/shared/leizhikai/train_internlm/configs/TARO_7B_v0_4_0.py',
        model_type='LLAMA',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
),

]
