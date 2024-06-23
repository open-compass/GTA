from mmengine.config import read_base
from copy import deepcopy
from opencompass.partitioners import SizePartitioner
from opencompass.models import HuggingFaceCausalLM
from opencompass.runners import LocalRunner
from opencompass.partitioners import SizePartitioner
from opencompass.tasks import OpenICLInferTask
from opencompass.models.lagent import LagentAgent
from lagent import ReAct
from lagent.agents.react import ReActProtocol
from opencompass.models.lagent import CodeAgent
from opencompass.lagent.actions.python_interpreter import PythonInterpreter
from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct

with read_base():
    # math baseline
    from .datasets.gsm8k.gsm8k_gen_d6de81 import gsm8k_datasets as gsm8k_baseline_datasets
    from .datasets.math.math_gen_1ed9c2 import math_datasets as math_baseline_datasets
    from .datasets.MathBench.mathbench_gen import mathbench_datasets as mathbench_baseline_datasets
    # math agent
    from .datasets.math.math_agent_gen_af2293 import math_datasets as math_agent_datasets
    from .datasets.gsm8k.gsm8k_agent_gen_be1606 import gsm8k_datasets as gsm8k_agent_datasets
    from .datasets.MathBench.mathbench_agent_gen_fbe13b import mathbench_agent_datasets
    # agent
    from .datasets.CIBench.CIBench_template_gen_e6b12a import cibench_datasets as cibench_template_datasets
    from .datasets.pluginEval.plugin_eval_v2_p10_gen_1ac254 import plugin_eval_datasets

    from .models.hf_internlm.hf_internlm_chat_7b import models as internlm_7b_chat_model
    from .models.qwen.hf_qwen_7b_chat import models as qwen_7b_chat_model

    from .internal.agent_prompts.cibench import FEWSHOT_INSTRUCTION, FORCE_STOP_PROMPT_EN, IPYTHON_INTERPRETER_DESCRIPTION
    from .internal.agent_prompts.math import MATH_SYSTEM_PROMPT
    
    from .summarizers.agent_bench import summarizer

datasets = []
datasets += gsm8k_baseline_datasets
datasets += math_baseline_datasets 
datasets += mathbench_baseline_datasets
datasets += gsm8k_agent_datasets
datasets += math_agent_datasets
datasets += mathbench_agent_datasets
datasets += cibench_template_datasets
datasets += plugin_eval_datasets


MATH_BASELINE_DATASET_NAMES = ['math_baseline_datasets', 'gsm8k_baseline_datasets', 'mathbench_baseline_datasets']
MATH_DATASET_NAMES = ['math_agent_datasets', 'gsm8k_agent_datasets', 'mathbench_agent_datasets']
CIBENCH_DATASET_NAMES = ['cibench_template_datasets']
PLUGINEVAL_DATASET_NAMES = ['plugin_eval_datasets']

_baseline_datasets = sum([v for k, v in locals().items() if k in MATH_BASELINE_DATASET_NAMES], [])
_baseline_models = sum([v for k, v in locals().items() if k.endswith("_model")], [])


agent_internlm = deepcopy(internlm_7b_chat_model[0])
agent_internlm['meta_template']['round'] = [
        dict(role="user", begin="<|User|>:", end="\n"),
        dict(role="assistant", begin="<|Bot|>:", end="<eoa>\n", generate=True),
        dict(role="system", begin="<|System|>:", end="\n"),
]
agent_qwen = deepcopy(qwen_7b_chat_model[0])
agent_qwen['meta_template']['round'] = [
        dict(role="user", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="assistant", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
        dict(role="system", begin="System response:", end="\n"),
]

_agent_models = [agent_internlm, agent_qwen]

# ---------------------------------------- MATH AGENT BEGIN ----------------------------------------
_math_agent_datasets = sum([v for k, v in locals().items() if k in MATH_DATASET_NAMES], [])

protocol = dict(
    type=ReActProtocol,
    action=dict(role="ACTION", begin="Tool:", end="\n"),
    action_input=dict(role="ARGS", begin="Tool Input:", end="\n"),
    finish=dict(role="FINISH", begin="FinalAnswer:", end="\n"),
    call_protocol=MATH_SYSTEM_PROMPT,
)

_math_agent_models = []
for m in _agent_models:
    m = deepcopy(m)
    origin_abbr = m.pop('abbr')
    abbr = origin_abbr + '-math-react'
    m.pop('batch_size', None)
    m.pop('max_out_len', None)
    m.pop('max_seq_len', None)
    run_cfg = m.pop('run_cfg', {})

    agent_model = dict(
        abbr=abbr,
        summarizer_abbr=origin_abbr,
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=3,
        llm=m,
        actions=[dict(type=PythonInterpreter)],
        protocol=protocol,
        batch_size=1,
        run_cfg=run_cfg,
    )
    _math_agent_models.append(agent_model)
# ---------------------------------------- MATH AGENT END ----------------------------------------

# ---------------------------------------- CIBENCH AGENT BEGIN ----------------------------------------
_cibench_agent_datasets = sum([v for k, v in locals().items() if k in CIBENCH_DATASET_NAMES], [])

protocol=dict(
    type=ReActProtocol,
    call_protocol=FEWSHOT_INSTRUCTION,
    force_stop=FORCE_STOP_PROMPT_EN,
    finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
)

_cibench_agent_models = []
for m in _agent_models:
    m = deepcopy(m)
    origin_abbr = m.pop('abbr')
    abbr = origin_abbr + '-cibench-react'
    m.pop('batch_size', None)
    m.pop('max_out_len', None)
    m.pop('max_seq_len', None)
    run_cfg = m.pop('run_cfg', {})

    agent_model = dict(
        abbr=abbr,
        summarizer_abbr=origin_abbr,
        type=CodeAgent,
        agent_type=CIReAct,
        max_turn=3,
        llm=m,
        actions=[dict(type=IPythonInterpreter, description=IPYTHON_INTERPRETER_DESCRIPTION)],
        protocol=protocol,
        batch_size=1,
        run_cfg=run_cfg,
    )
    _cibench_agent_models.append(agent_model)
# ---------------------------------------- CIBENCH AGENT END ----------------------------------------

# ---------------------------------------- PLUGIN EVAL BEGIN ----------------------------------------
_plugin_eval_datasets = sum([v for k, v in locals().items() if k in PLUGINEVAL_DATASET_NAMES], [])
_plugin_eval_models = []
for m in _baseline_models:
    m = deepcopy(m)
    m['summarizer_abbr'] = m['abbr']
    m['abbr'] = m['abbr'] + '--plugin-eval'
    _plugin_eval_models.append(m)

model_dataset_combinations, models, datasets = [], [], []
if _baseline_datasets:
    model_dataset_combinations.append(dict(models=_baseline_models, datasets=_baseline_datasets))
    models.extend(_baseline_models)
    datasets.extend(_baseline_datasets)
if _math_agent_datasets:
    model_dataset_combinations.append(dict(models=_math_agent_models, datasets=_math_agent_datasets))
    models.extend(_math_agent_models)
    datasets.extend(_math_agent_datasets)
if _cibench_agent_datasets:
    model_dataset_combinations.append(dict(models=_cibench_agent_models, datasets=_cibench_agent_datasets))
    models.extend(_cibench_agent_models)
    datasets.extend(_cibench_agent_datasets)
if _plugin_eval_datasets:
    model_dataset_combinations.append(dict(models=_plugin_eval_models, datasets=_plugin_eval_datasets))
    models.extend(_plugin_eval_models)
    datasets.extend(_plugin_eval_datasets)

from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=100000),
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
