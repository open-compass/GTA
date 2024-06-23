from mmengine.config import read_base

with read_base():

    from ...datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from ...datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from ...datasets.ceval.ceval_internal_gen_2daf24 import ceval_datasets
    from ...datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from ...datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from ...datasets.nq.nq_gen_c788f6 import nq_datasets
    from ...datasets.race.race_gen_69ee4f import race_datasets
    from ...datasets.winogrande.winogrande_gen_a9ede5 import winogrande_datasets
    from ...datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from ...datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen_c68933 import cluewsc_datasets
    from ...datasets.FewCLUE_csl.FewCLUE_csl_gen_28b223 import csl_datasets
    from ...datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ...datasets.math.math_evaluatorv2_gen_265cce import math_datasets
    from ...datasets.TheoremQA.TheoremQA_gen_7009de import TheoremQA_datasets
    from ...datasets.humaneval.humaneval_gen_6d1cc2 import humaneval_datasets
    # TODO: using sanitized_mbpp_datasets
    from ...datasets.mbpp.mbpp_gen_caa7ab import mbpp_datasets

    # agent
    from ...datasets.math.math_agent_evaluatorv2_gen_0c1b4e import math_datasets as math_agent_datasets
    # from ...datasets.gsm8k.gsm8k_agent_gen_1f182e import gsm8k_datasets as gsm8k_agent_datasets


MATH_AGENT_DATASET_NAMES = [
    'math_agent_datasets',
    'gsm8k_agent_datasets',
    'mathbench_agent_datasets'
]

LONGTEXT_DATASET_NAMES = [
    'longeval_datasets', 
    'longbench_datasets', 
    'leval_datasets'
]

CIBENCH_DATASET_NAMES = [
    'cibench_agent_datasets'
]

PLUGINEVAL_DATASET_NAMES = [
    'plugin_eval_datasets'
]

BLACKLIST_DATASET_NAMES = [
    # 'math_agent_datasets'
]

DATASETS_BLACKLIST = MATH_AGENT_DATASET_NAMES + LONGTEXT_DATASET_NAMES + CIBENCH_DATASET_NAMES + \
    PLUGINEVAL_DATASET_NAMES

# base datasets
base_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in DATASETS_BLACKLIST and k not in BLACKLIST_DATASET_NAMES), [])
# longtext datasets
longtext_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in LONGTEXT_DATASET_NAMES and k not in BLACKLIST_DATASET_NAMES), [])
# math agent datasets
math_agent_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in MATH_AGENT_DATASET_NAMES and k not in BLACKLIST_DATASET_NAMES), [])
# cibench datasets
cibench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in CIBENCH_DATASET_NAMES and k not in BLACKLIST_DATASET_NAMES), [])
# plugin eval datasets
plugin_eval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in PLUGINEVAL_DATASET_NAMES and k not in BLACKLIST_DATASET_NAMES), [])
