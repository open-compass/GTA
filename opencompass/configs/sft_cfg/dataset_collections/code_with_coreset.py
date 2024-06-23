from mmengine.config import read_base

with read_base():
    # 学科
    from ...datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from ...datasets.agieval.agieval_gen_397d81 import agieval_datasets
    from ...datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from ...datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
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
    from ...datasets.math.math_gen_265cce import math_datasets
    from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # 代码
    # the new prompt can get better performance than the old one
    from ...datasets.mbpp.mbpp_gen_caa7ab import mbpp_datasets
    from ...datasets.humaneval.humaneval_gen_6d1cc2 import humaneval_datasets
    # old prompt setting
    # from ...datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ...datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    # 理解
    # 阅读理解
    from ...datasets.obqa.obqa_gen_9069e4 import obqa_datasets
    # 内容总结
    from ...datasets.Xsum.Xsum_gen_31397e import Xsum_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
