from lagent.agents.react import ReActProtocol
from mmengine.config import read_base

from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct
from opencompass.models.lagent import CodeAgent
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models.internal import InternLMwithModule

with read_base():
    from .datasets.CIBench.CIBench_template_gen_e6b12a import \
        cibench_datasets as datasets
    from .summarizers.cibench import summarizer

FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""

FEWSHOT_INSTRUCTION = """\
You are an assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.

For example:

File url: `xxxx`
### Step 1. Load the dataset from the url into a pandas DataFrame named `df`.

{thought} We should use `pandas` to solve this step.
{action} IPythonInterpreter
{action_input} ```python
import pandas as pd
url = "xxxx"
data = pd.read_csv(url)
```
{response} The code is succeed without any outputs.

Let us begin from here!
"""

IPYTHON_INTERPRETER_DESCRIPTION = '''\
It can run Python code in a manner as jupyter notebook. The code must be a valid code that contains only python method.'''

_meta_template = dict(
    round=[
        dict(role="user", begin="<|User|>:", end="\n"),
        dict(role="assistant", begin="<|Bot|>:", end="<eoa>\n", generate=True),
        dict(role="system", begin="<|System|>:", end="\n"),
    ],
    eos_token_id=103028,
)

_meta_template_70 = dict(
    round=[
        dict(role="user", begin="<|Human|>െ", end="െ\n "),
        dict(role="assistant", begin="<|Assistant|>െ", end="ി\n ", generate=True),
        dict(role="system", begin="<|System|>െ", end="\n "),

    ],
    eos_token_id=45623,
)

_meta_template_123 = dict(
    round=[
        dict(role="user", begin="<|Human|>െ", end="െ\n "),
        dict(role="assistant", begin="<|Assistant|>െ", end="ി\n ", generate=True),
        dict(role="system", begin="<|System|>െ", end="\n "),
    ],
    eos_token_id=45623,
)

actions=[dict(type=IPythonInterpreter, user_data_dir='./data/cibench_dataset/datasources',
                 description=IPYTHON_INTERPRETER_DESCRIPTION)]
protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        )
models = [
    dict(
        abbr="internlm-chat-7b-hf-v11",
        type=CodeAgent,
        agent_type=CIReAct,
        llm=dict(
            type=HuggingFaceCausalLM,
            path="internlm/internlm-chat-7b-v1_1",
            tokenizer_path="internlm/internlm-chat-7b-v1_1",
            tokenizer_kwargs=dict(
                padding_side="left",
                truncation_side="left",
                use_fast=False,
                trust_remote_code=True,
            ),
            max_seq_len=2048,
            end_str='<eoa>',
            meta_template=_meta_template,
            model_kwargs=dict(trust_remote_code=True, device_map="auto"),
        ),
        protocol=protocol,
        actions=actions,
        max_turn=3,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=8,
    ),
    dict(
        abbr="internlm-chat-20b-hf",
        type=CodeAgent,
        agent_type=CIReAct,
        llm=dict(
            type=HuggingFaceCausalLM,
            path="internlm/internlm-chat-20b",
            tokenizer_path="internlm/internlm-chat-20b",
            tokenizer_kwargs=dict(
                padding_side="left",
                truncation_side="left",
                use_fast=False,
                trust_remote_code=True,
            ),
            max_seq_len=2048,
            end_str='<eoa>',
            meta_template=_meta_template,
            model_kwargs=dict(trust_remote_code=True, device_map="auto"),
        ),
        protocol=protocol,
        actions=actions,
        max_turn=3,
        run_cfg=dict(num_gpus=2, num_procs=1),
        batch_size=8,
    ),
    dict(
        abbr="InternLM-70B-Chat",
        type=CodeAgent,
        agent_type=CIReAct,
        llm=dict(
            type=InternLMwithModule,
            model_type="LLAMA",
            path="/mnt/petrelfs/share_data/zhoufengzhe/weights/shlab/Euclid_70B_2.0.0_FT_0.11/1550/",
            module_path="/mnt/petrelfs/share_data/zhoufengzhe/modules/train_internlm--b3d10bd2b04535ec0537e7659ee1f6f8b8ef2f55",
            model_config="/mnt/petrelfs/share_data/zhoufengzhe/modules/train_internlm--b3d10bd2b04535ec0537e7659ee1f6f8b8ef2f55/configs/Euclid_70B_1.2.0.py",
            tokenizer_path="/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model",
            tokenizer_type="v4",
            meta_template=_meta_template_70,
            max_seq_len=2048,
            sync_rank=True,
        ),
        actions=actions,
        protocol=protocol,
        max_turn=3,
        batch_size=8,
        run_cfg=dict(num_gpus=4, num_procs=4),
    ),
    dict(
        abbr="InternLM-123B-Chat",
        type=CodeAgent,
        agent_type=CIReAct,
        llm=dict(
            type=InternLMwithModule,
            model_type="LLAMA",
            path="/mnt/petrelfs/share_data/zhoufengzhe/weights/shlab/sft_123b_chat_enhance_replace_moss/1110/",
            module_path="/mnt/petrelfs/share_data/zhoufengzhe/modules/train_internlm--3be2e794e9df3fd799e4987d37d9f6d453bcc315",
            model_config="/mnt/petrelfs/share_data/zhoufengzhe/modules/train_internlm--3be2e794e9df3fd799e4987d37d9f6d453bcc315/configs/plato_123b_8k_sft.py",
            tokenizer_path="/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model",
            tokenizer_type="v4",
            meta_template=_meta_template_123,
            max_seq_len=2048,
            sync_rank=True,
        ),
        actions=actions,
        protocol=protocol,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=8),
    ),
]


from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=200),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=256,
        partition="llmeval",
        quotatype="reserved",
        extra_command=['-c 12'],
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=256,
        partition="llmeval",
        quotatype="reserved",
        task=dict(type=OpenICLEvalTask),
    ),
)
