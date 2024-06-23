from mmengine.config import read_base

with read_base():
    from ..datasets.CLUE_CMRC.CLUE_CMRC_gen_8484b9 import CMRC_datasets
    from ..datasets.CLUE_DRCD.CLUE_DRCD_gen_8484b9 import DRCD_datasets
    from ..datasets.CLUE_afqmc.CLUE_afqmc_ppl_7b0c1e import afqmc_datasets
    from ..datasets.FewCLUE_bustm.FewCLUE_bustm_ppl_9ef540 import bustm_datasets
    from ..datasets.FewCLUE_chid.FewCLUE_chid_ppl_acccb5 import chid_datasets
    from ..datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_ppl_4284a0 import cluewsc_datasets
    from ..datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_ppl_1ce587 import eprstmt_datasets
    from ..datasets.humaneval.humaneval_gen_fd5822 import humaneval_datasets
    from ..datasets.mbpp.mbpp_gen_6590b0 import mbpp_datasets
    from ..datasets.lambada.lambada_gen_8b48a5 import lambada_datasets
    from ..datasets.storycloze.storycloze_ppl_afd16f import storycloze_datasets
    from ..datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_ppl_0748aa import AX_b_datasets
    from ..datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_ppl_50f8f6 import AX_g_datasets
    from ..datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_9619db import BoolQ_datasets
    from ..datasets.SuperGLUE_CB.SuperGLUE_CB_ppl_11c175 import CB_datasets
    from ..datasets.SuperGLUE_COPA.SuperGLUE_COPA_ppl_54058d import COPA_datasets
    from ..datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_ppl_866273 import MultiRC_datasets
    from ..datasets.SuperGLUE_RTE.SuperGLUE_RTE_ppl_50f8f6 import RTE_datasets
    from ..datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_0f7784 import ReCoRD_datasets
    from ..datasets.SuperGLUE_WiC.SuperGLUE_WiC_ppl_3fb6fd import WiC_datasets
    from ..datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl_f37e78 import WSC_datasets
    from ..datasets.race.race_ppl_abed12 import race_datasets
    from ..datasets.math.math_gen_559593 import math_datasets
    from ..datasets.gsm8k.gsm8k_gen_1dce88 import gsm8k_datasets
    from ..datasets.summedits.summedits_ppl_fa58ba import summedits_datasets
    from ..datasets.hellaswag.hellaswag_ppl_9dbb12 import hellaswag_datasets
    from ..datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    from ..datasets.winogrande.winogrande_ppl_9307fd import winogrande_datasets
    from ..datasets.obqa.obqa_ppl_1defe8 import obqa_datasets
    from ..datasets.nq.nq_gen_2463e2 import nq_datasets
    from ..datasets.triviaqa.triviaqa_gen_429db5 import triviaqa_datasets
    from ..datasets.crowspairs.crowspairs_ppl_47f211 import crowspairs_datasets
    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets
    from ..datasets.agieval.agieval_mixed import agieval_datasets
    from ..datasets.ceval.ceval_ppl_578f8d import ceval_datasets
    from ..datasets.bbh.bbh_gen_6bd693 import bbh_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
