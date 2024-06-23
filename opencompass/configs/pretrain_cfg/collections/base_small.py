from mmengine.config import read_base

with read_base():
    from ...datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from ...datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    from ...datasets.CLUE_afqmc.CLUE_afqmc_ppl_7b0c1e import afqmc_datasets
    from ...datasets.FewCLUE_bustm.FewCLUE_bustm_ppl_e53034 import bustm_datasets
    from ...datasets.FewCLUE_chid.FewCLUE_chid_ppl_8f2872 import chid_datasets
    from ...datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_ppl_868415 import cluewsc_datasets
    from ...datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_ppl_f1e631 import eprstmt_datasets
    from ...datasets.humaneval.humaneval_gen_a82cae import humaneval_datasets
    from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from ...datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from ...datasets.storycloze.storycloze_ppl_496661 import storycloze_datasets
    from ...datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_ppl_6db806 import AX_b_datasets
    from ...datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_ppl_66caf3 import AX_g_datasets
    from ...datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314b96 import BoolQ_datasets
    from ...datasets.SuperGLUE_CB.SuperGLUE_CB_ppl_0143fe import CB_datasets
    from ...datasets.SuperGLUE_COPA.SuperGLUE_COPA_ppl_9f3618 import COPA_datasets
    from ...datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_ppl_ced824 import MultiRC_datasets
    from ...datasets.SuperGLUE_RTE.SuperGLUE_RTE_ppl_66caf3 import RTE_datasets
    from ...datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    from ...datasets.SuperGLUE_WiC.SuperGLUE_WiC_ppl_312de9 import WiC_datasets
    from ...datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl_d0f531 import WSC_datasets
    from ...datasets.race.race_ppl_a138cd import race_datasets
    from ...datasets.math.math_gen_265cce import math_datasets
    from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ...datasets.summedits.summedits_ppl_1fbeb6 import summedits_datasets
    from ...datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets
    from ...datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    from ...datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets
    from ...datasets.obqa.obqa_ppl_c7c154 import obqa_datasets
    from ...datasets.nq.nq_gen_c788f6 import nq_datasets
    from ...datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from ...datasets.crowspairs.crowspairs_ppl_e811e1 import crowspairs_datasets
    from ...datasets.mmlu.mmlu_ppl import mmlu_datasets
    from ...datasets.agieval.agieval_mixed import agieval_datasets
    from ...datasets.ceval.ceval_ppl_578f8d import ceval_datasets
    from ...datasets.bbh.bbh_gen_5b92b0 import bbh_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
