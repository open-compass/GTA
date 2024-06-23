from mmengine.config import read_base

with read_base():
    # --------- 语言 Language ---------
    from ...datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from ...datasets.summedits.summedits_gen_315438 import summedits_datasets
    from ...datasets.FewCLUE_chid.FewCLUE_chid_gen_0a29a2 import chid_datasets
    from ...datasets.CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets
    from ...datasets.FewCLUE_bustm.FewCLUE_bustm_gen_634f41 import bustm_datasets
    from ...datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen_c68933 import cluewsc_datasets
    from ...datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
    from ...datasets.winogrande.winogrande_gen_a9ede5 import winogrande_datasets
    from ...datasets.flores.flores_gen_806ede import flores_datasets
    from ...datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    from ...datasets.CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets
    from ...datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from ...datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    from ...datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_gen_27071f import MultiRC_datasets
    from ...datasets.race.race_gen_69ee4f import race_datasets
    from ...datasets.obqa.obqa_gen_9069e4 import obqa_datasets
    from ...datasets.drop.drop_gen_8a9ed9 import drop_datasets
    from ...datasets.FewCLUE_csl.FewCLUE_csl_gen_28b223 import csl_datasets
    from ...datasets.lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    from ...datasets.Xsum.Xsum_gen_31397e import Xsum_datasets
    from ...datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_gen_740ea0 import eprstmt_datasets
    from ...datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from ...datasets.FewCLUE_tnews.FewCLUE_tnews_gen_b90e4a import tnews_datasets

    # --------- 知识 Knowledge ---------
    from ...datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    from ...datasets.commonsenseqa.commonsenseqa_gen_c946f2 import commonsenseqa_datasets
    from ...datasets.nq.nq_gen_c788f6 import nq_datasets
    from ...datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    # different with v052
    from ...datasets.ceval.ceval_internal_gen_2daf24 import ceval_datasets
    from ...datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from ...datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from ...datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from ...datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from ...datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from ...datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
    from ...datasets.wikibench.wikibench_gen_f96ece import wikibench_datasets
    from ...datasets.commonsenseqa_cn.commonsenseqacn_gen_d380d0 import commonsenseqacn_datasets
    from ...datasets.nq_cn.nqcn_gen_141737 import nqcn_datasets

    # --------- 推理 Reasoning ---------
    from ...datasets.CLUE_cmnli.CLUE_cmnli_gen_1abf97 import cmnli_datasets
    from ...datasets.CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets
    from ...datasets.FewCLUE_ocnli_fc.FewCLUE_ocnli_fc_gen_f97a97 import ocnli_fc_datasets
    from ...datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen_4dfefa import AX_b_datasets
    from ...datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_gen_68aac7 import AX_g_datasets
    from ...datasets.SuperGLUE_RTE.SuperGLUE_RTE_gen_68aac7 import RTE_datasets
    from ...datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    from ...datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from ...datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from ...datasets.siqa.siqa_gen_e78df3 import siqa_datasets
    from ...datasets.TheoremQA.TheoremQA_gen_7009de import TheoremQA_datasets
    from ...datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from ...datasets.strategyqa.strategyqa_gen_1180a7 import strategyqa_datasets

    # --------- 数学 Mathematics ---------
    # math-evaluator v1
    # from ...datasets.math.math_gen_265cce import math_datasets
    # math-evaluator v2
    from ...datasets.math.math_evaluatorv2_gen_265cce import math_datasets
    from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # NOTE: the setting of mathbench_gen_ad37c1 dataset is not exists in the common dataset, add this in sft_cfg
    # from ..datasets.MathBench.mathbench_gen_ad37c1 import mathbench_datasets
    # use CoT in MathBench, and update prompt
    from ...datasets.MathBench.mathbench_cot_gen_66f329 import mathbench_datasets
    from ...datasets.MathBench.mathbench_arith_gen_ccd638 import mathbench_datasets as arithmath_datasets
    from ...datasets.gsm_hard.gsmhard_gen_8a1400 import gsmhard_datasets
    # NOTE: the setting of gsm8k_option_gen_108724 dataset is not exists in the common dataset, add this in sft_cfg
    from ..datasets.gsm8k_extra.gsm8k_option_gen_108724 import gsm8k_option_datasets as gsm8k_option_datasets

    # --------- 代码 Coding ---------
    # the new prompt can get better performance than the old one
    from ...datasets.mbpp.mbpp_gen_caa7ab import mbpp_datasets
    from ...datasets.humaneval.humaneval_gen_6d1cc2 import humaneval_datasets
    # old prompt setting
    # from ...datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from ...datasets.ds1000.ds1000_service_eval_gen_cbc84f import ds1000_datasets
    from ...datasets.py150.py150_gen_38b13d import py150_datasets
    from ...datasets.clozeTest_maxmin.clozeTest_maxmin_gen_c205fb import maxmin_datasets

    # --------- longeval ---------
    from ...datasets.longeval.longeval import longeval_datasets

    # --------- Math Agent ---------
    # math-evaluator v1
    # from ...datasets.math.math_agent_gen_861b4f import math_datasets as math_agent_datasets
    # math-evaluator v2
    # from ...datasets.math.math_agent_evaluatorv2_gen_861b4f import math_datasets as math_agent_datasets
    # fix math-agent prompt error
    from ...datasets.math.math_agent_evaluatorv2_gen_0c1b4e import math_datasets as math_agent_datasets

    # fix gsm8k-agent prompt error
    # from ...datasets.gsm8k.gsm8k_agent_gen_3ac57d import gsm8k_datasets as gsm8k_agent_datasets
    from ...datasets.gsm8k.gsm8k_agent_gen_1f182e import gsm8k_datasets as gsm8k_agent_datasets

    # fix mathbench-agent prompt error and config BC
    # from ..datasets.MathBench.mathbench_agent_gen_568903 import mathbench_agent_datasets
    from ...datasets.MathBench.mathbench_agent_gen_48ec47 import mathbench_agent_datasets

    # --------- CIBench ---------
    # NOTE: the setting of CIBench_gen_8ab0dc change the evaluator logic in the common dataset, add this in sft_cfg
    from ..datasets.CIBench.CIBench_gen_8ab0dc import cibench_datasets as cibench_agent_datasets

    # --------- plugin_eval ---------
    from ...datasets.pluginEval.plugin_eval_v2_gen_1ac254 import plugin_eval_datasets
    # add subjective QA
    from ..subjective_qa.subjectiveqav3_gen import subjectiveqav3_datasets

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
