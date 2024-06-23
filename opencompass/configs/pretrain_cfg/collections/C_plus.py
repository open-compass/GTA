from mmengine.config import read_base

with read_base():
    from ..datasets.FewCLUE_bustm.FewCLUE_bustm_ppl_9ef540 import bustm_datasets
    from ..datasets.CLUE_afqmc.CLUE_afqmc_ppl_7b0c1e import afqmc_datasets
    from ..datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_ppl_1ce587 import eprstmt_datasets
    from ..datasets.FewCLUE_ocnli_fc.FewCLUE_ocnli_fc_ppl_c08300 import ocnli_fc_datasets
    from ..datasets.CLUE_ocnli.CLUE_ocnli_ppl_98dd6e import ocnli_datasets
    from ..datasets.CLUE_cmnli.CLUE_cmnli_ppl_98dd6e import cmnli_datasets
    from ..datasets.FewCLUE_csl.FewCLUE_csl_ppl_841b62 import csl_datasets
    from ..datasets.FewCLUE_chid.FewCLUE_chid_ppl_acccb5 import chid_datasets
    from ..datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_ppl_4284a0 import cluewsc_datasets
    from ..datasets.FewCLUE_tnews.FewCLUE_tnews_ppl_7d1c07 import tnews_datasets
    from ..datasets.CLUE_C3.CLUE_C3_ppl_56b537 import C3_datasets
    from ..datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from ..datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    from ..datasets.lcsts.lcsts_gen_9b0b89 import lcsts_datasets

    from ..datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    from ..datasets.commonsenseqa.commonsenseqa_ppl_716f78 import commonsenseqa_datasets
    from ..datasets.gsm8k.gsm8k_gen_1dce88 import gsm8k_datasets
    from ..datasets.humaneval.humaneval_gen_fd5822 import humaneval_datasets
    from ..datasets.mbpp.mbpp_gen_6590b0 import mbpp_datasets
    from ..datasets.triviaqa.triviaqa_gen_429db5 import triviaqa_datasets
    from ..datasets.nq.nq_gen_c788f6 import nq_datasets

    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets
    from ..datasets.agieval.agieval_mixed import agieval_datasets
    from ..datasets.ceval.ceval_ppl_578f8d import ceval_datasets

    from ..datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_ppl_0748aa import AX_b_datasets
    from ..datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_ppl_50f8f6 import AX_g_datasets
    from ..datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_9619db import BoolQ_datasets
    from ..datasets.SuperGLUE_CB.SuperGLUE_CB_ppl_11c175 import CB_datasets
    from ..datasets.SuperGLUE_COPA.SuperGLUE_COPA_ppl_54058d import COPA_datasets
    from ..datasets.SuperGLUE_RTE.SuperGLUE_RTE_ppl_50f8f6 import RTE_datasets
    from ..datasets.SuperGLUE_WiC.SuperGLUE_WiC_ppl_3fb6fd import WiC_datasets
    from ..datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl_f37e78 import WSC_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
