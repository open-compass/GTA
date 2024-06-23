from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from ..datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from ..datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from ..datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from ..datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    from ..datasets.CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets
    from ..datasets.FewCLUE_bustm.FewCLUE_bustm_gen_634f41 import bustm_datasets
    from ..datasets.FewCLUE_chid.FewCLUE_chid_gen_0a29a2 import chid_datasets
    from ..datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen_c68933 import cluewsc_datasets
    from ..datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_gen_740ea0 import eprstmt_datasets
    from ..datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from ..datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from ..datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from ..datasets.storycloze.storycloze_gen_7f656a import storycloze_datasets
    from ..datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen_4dfefa import AX_b_datasets
    from ..datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_gen_68aac7 import AX_g_datasets
    from ..datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    from ..datasets.SuperGLUE_CB.SuperGLUE_CB_gen_854c6c import CB_datasets
    from ..datasets.SuperGLUE_COPA.SuperGLUE_COPA_gen_91ca53 import COPA_datasets
    from ..datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_gen_27071f import MultiRC_datasets
    from ..datasets.SuperGLUE_RTE.SuperGLUE_RTE_gen_68aac7 import RTE_datasets
    from ..datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    from ..datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from ..datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_8a881c import WSC_datasets
    from ..datasets.race.race_gen_69ee4f import race_datasets
    from ..datasets.math.math_gen_265cce import math_datasets
    from ..datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ..datasets.summedits.summedits_gen_315438 import summedits_datasets
    from ..datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from ..datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from ..datasets.winogrande.winogrande_gen_a9ede5 import winogrande_datasets
    from ..datasets.obqa.obqa_gen_9069e4 import obqa_datasets
    from ..datasets.nq.nq_gen_c788f6 import nq_datasets
    from ..datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from ..datasets.crowspairs.crowspairs_gen_21f7cb import crowspairs_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
