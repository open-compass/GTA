from mmengine.config import read_base
from copy import deepcopy
from opencompass.partitioners import SizePartitioner
from opencompass.models import HuggingFaceCausalLM
from opencompass.runners import LocalRunner
from opencompass.partitioners import SizePartitioner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.humaneval_cn.humaneval_cn_gen_6313aa import humaneval_cn_datasets
    from .datasets.humanevalx.humanevalx_gen_620cfa import humanevalx_datasets
    from .datasets.humaneval_plus.humaneval_plus_gen_8e312c import humaneval_plus_datasets
    from .datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from .datasets.mbpp_cn.mbpp_cn_gen_1d1481 import mbpp_cn_datasets
    from .datasets.mbpp.sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets

    from .datasets.humaneval.humaneval_passk_gen_8e312c import humaneval_datasets as humaneval_passk_datasets
    from .datasets.humaneval_cn.humaneval_cn_passk_gen_6313aa import humaneval_cn_datasets as humaneval_cn_passk_datasets
    from .datasets.humaneval_plus.humaneval_plus_passk_gen_8e312c import humaneval_plus_datasets as humaneval_plus_passk_datasets
    from .datasets.mbpp.mbpp_passk_gen_1e1056 import mbpp_datasets as mbpp_passk_datasets
    from .datasets.mbpp_cn.mbpp_cn_passk_gen_1d1481 import mbpp_cn_datasets as mbpp_cn_passk_datasets
    from .datasets.mbpp.sanitized_mbpp_passk_gen_1e1056 import sanitized_mbpp_datasets as sanitized_mbpp_passk_datasets

    from .datasets.humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets as humaneval_repeat10_datasets
    from .datasets.humaneval_cn.humaneval_cn_repeat10_gen_6313aa import humaneval_cn_datasets as humaneval_cn_repeat10_datasets
    from .datasets.humaneval_plus.humaneval_plus_repeat10_gen_8e312c import humaneval_plus_datasets as humaneval_plus_repeat10_datasets
    from .datasets.mbpp.mbpp_repeat10_gen_1e1056 import mbpp_datasets as mbpp_repeat10_datasets
    from .datasets.mbpp_cn.mbpp_cn_repeat10_gen_1d1481 import mbpp_cn_datasets as mbpp_cn_repeat10_datasets
    from .datasets.mbpp.sanitized_mbpp_repeat10_gen_1e1056 import sanitized_mbpp_datasets as sanitized_mbpp_repeat10_datasets

    from .models.hf_internlm.hf_internlm_chat_7b import models as internlm_7b_chat_model
    from .models.qwen.hf_qwen_7b_chat import models as qwen_7b_chat_model
    from .models.chatglm.hf_chatglm3_6b import models as chatglm3_6b_chat_model
    from .models.hf_internlm.hf_internlm_7b import models as internlm_7b_base_model
    from .models.qwen.hf_qwen_7b import models as qwen_7b_base_model
    from .models.chatglm.hf_chatglm3_6b_base import models as chatglm3_6b_base_model

    from .summarizers.code_passk import summarizer
# pass1
base_datasets = []
base_datasets += humaneval_datasets
base_datasets += humaneval_cn_datasets
base_datasets += humanevalx_datasets
base_datasets += humaneval_plus_datasets
base_datasets += mbpp_datasets
base_datasets += mbpp_cn_datasets
base_datasets += sanitized_mbpp_datasets

base_models = sum([v for k, v in locals().items() if k.endswith("_model")], [])

# pass10
passk_datasets = []
passk_datasets += humaneval_passk_datasets
passk_datasets += humaneval_cn_passk_datasets
passk_datasets += humaneval_plus_passk_datasets
passk_datasets += mbpp_passk_datasets
passk_datasets += mbpp_cn_passk_datasets
passk_datasets += sanitized_mbpp_passk_datasets

qwen_7b_chat_passk_model = deepcopy(qwen_7b_chat_model[0])
qwen_7b_base_passk_model = deepcopy(qwen_7b_base_model[0])
chatglm3_6b_base_passk_model = deepcopy(chatglm3_6b_base_model[0])
passk_models = [v for k, v in locals().items() if k.endswith("passk_model")]
for _model in passk_models:
    _model['generation_kwargs'] = dict(
            num_return_sequences=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

# pass10 only for internlm
repeat10_datasets = []
repeat10_datasets += humaneval_repeat10_datasets
repeat10_datasets += humaneval_cn_repeat10_datasets
repeat10_datasets += humaneval_plus_repeat10_datasets
repeat10_datasets += mbpp_repeat10_datasets
repeat10_datasets += mbpp_cn_repeat10_datasets
repeat10_datasets += sanitized_mbpp_repeat10_datasets


# chatglm3 cannot return multiple outputs
chatglm3_6b_chat_repeat10_model = deepcopy(chatglm3_6b_chat_model[0])
internlm_7b_chat_repeat10_model = deepcopy(internlm_7b_chat_model[0])
internlm_7b_base_repeat10_model = deepcopy(internlm_7b_base_model[0])
repeat10_models = [v for k, v in locals().items() if k.endswith("repeat10_model")]
for _model in repeat10_models:
    _model['generation_kwargs'] = dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

# all running combs
model_dataset_combinations = [
    dict(models=base_models, datasets=base_datasets),
    dict(models=passk_models, datasets=passk_datasets),
    dict(models=repeat10_models, datasets=repeat10_datasets),
]

# This union of models and datasets in model_dataset_combinations should be
# stored in the `models` and `datasets` variables below. Otherwise, modules
# like summarizer will miss out some information.
models = [*base_models]
datasets = [*base_datasets, *passk_datasets, *repeat10_datasets]


from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=10000),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=256,
        partition="llmeval",
        quotatype="reserved",
        retry=1,
        extra_command=['-c 12'],
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=256,
        partition="llmeval",
        quotatype="reserved",
        retry=0,
        task=dict(type=OpenICLEvalTask),
    ),
)

