from mmengine.config import read_base

with read_base():
    from ..groups.agieval import agieval_summary_groups
    from ..groups.mmlu import mmlu_summary_groups
    from ..groups.ceval import ceval_summary_groups
    from ..groups.bbh import bbh_summary_groups
    from ..groups.GaokaoBench import GaokaoBench_summary_groups
    from ..groups.flores import flores_summary_groups


summarizer = dict(
    dataset_abbrs = [
        # 中文综合能力
        'chid-dev',
        'chid-test',
        'cluewsc-dev',
        'cluewsc-test',
        'afqmc-dev',
        'afqmc-test',
        'eprstmt-dev',
        'eprstmt-test',
        # 代码能力
        'openai_humaneval',
        'mbpp',
        # 阅读能力
        'race-middle',
        'race-high',
        # 推理能力
        'gsm8k_main',
        'math',
        # 安全
        'crows_pairs',
        'allenai_real-toxicity-prompts',
        'truthful_qa',
        'triviaqa',
        'nq',
        'CMRC_dev',
        'CMRC_test',
        'DRCD_dev',
        'DRCD_test',
        'csl_dev',
        'csl_test',
        # CEval
        "ceval-stem",
        "ceval-social-science",
        "ceval-humanities",
        "ceval-other",
        "ceval",
        "ceval-hard",
        # MMLU
        'mmlu-humanities',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-other',
        'mmlu-all-set',
        'mmlu-weighted-all-set',
        # AGIEval
        'agieval-gaokao-full',
        'agieval',
        "GaokaoBench",
        # flores
        'flores_100_Indo-European-Germanic_English',
        'flores_100_English_Indo-European-Germanic',
        'flores_100_Indo-European-Romance_English',
        'flores_100_English_Indo-European-Romance',
        "flores_100_zho-eng",
        "flores_100_eng-zho",
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore')
)
