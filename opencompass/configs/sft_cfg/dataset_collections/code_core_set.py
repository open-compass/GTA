from mmengine.config import read_base

with read_base():
    # 学科
    from ...datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from ...datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # 语言
    # 成语习语
    from ...datasets.FewCLUE_chid.FewCLUE_chid_gen_0a29a2 import chid_datasets
    # 指代消解
    from ...datasets.winogrande.winogrande_gen_a9ede5 import winogrande_datasets
    # 知识
    # 知识问答
    from ...datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    from ...datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    # 推理
    # 文本蕴含
    from ...datasets.CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets
    # 常识
    from ...datasets.piqa.piqa_gen_1194eb import piqa_datasets
    # 数学
    from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ...datasets.math.math_gen_265cce import math_datasets
    # 代码
    # the new prompt can get better performance than the old one
    from ...datasets.mbpp.mbpp_gen_caa7ab import mbpp_datasets
    from ...datasets.humaneval.humaneval_gen_6d1cc2 import humaneval_datasets
    # old prompt setting
    # from ...datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    # 综合推理
    from ...datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    # 理解
    # 阅读理解
    from ...datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from ...datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    # 相似度
    from ...datasets.CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets
    # 内容总结
    from ...datasets.Xsum.Xsum_gen_31397e import Xsum_datasets
    # 新的代码数据
    from ...datasets.ds1000.ds1000_service_eval_gen_cbc84f import ds1000_datasets
    from ...datasets.clozeTest_maxmin.clozeTest_maxmin_gen_c205fb import maxmin_datasets
    from ...datasets.py150.py150_gen_38b13d import py150_datasets
    # humanevalx need docker during eval, not fully support now
    # from ...datasets.humanevalx.humanevalx_gen_0af626 import humanevalx_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
