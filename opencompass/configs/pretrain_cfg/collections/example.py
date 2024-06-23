from mmengine.config import read_base

with read_base():
    # from ...datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    # from ...datasets.triviaqa.triviaqa_gen_0356ec import triviaqa_datasets
    # from ...datasets.nq.nq_gen_0356ec import nq_datasets
    from ...datasets.humaneval.humaneval_gen_fd5822 import humaneval_datasets
    # from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    # from ...datasets.TheoremQA.TheoremQA_gen_424e0a import TheoremQA_datasets
    # from ...datasets.math.math_gen_265cce import math_datasets
    # from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    #
    # from ...datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    # from ...datasets.bbh.bbh_gen_6bd693 import bbh_datasets
    # from ...datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    # from ...datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets

    # from ...datasets.ceval.ceval_ppl_578f8d import ceval_datasets
    # from ...datasets.chinese_language_base.chinese_language_base_abc import chinese_language_base_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
