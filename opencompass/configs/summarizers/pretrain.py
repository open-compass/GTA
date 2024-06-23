from mmengine.config import read_base
from opencompass.summarizers import SubjectiveSummarizer, PretrainSummarizer

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.flores import flores_summary_groups
    from .groups.tydiqa import tydiqa_summary_groups
    from .groups.xiezhi import xiezhi_summary_groups
    from .groups.pretrain import chinese_summary_groups, coding_summary_groups, \
        common_qa_summary_groups, english_summary_groups, reasoning_summary_groups, \
        fact_qa_summary_groups, race_summary_groups, completion_summary_groups

summarizer = dict(
    type=PretrainSummarizer,
    dataset_abbrs=[
        # '--- ChineseUniversal ---',
        'CMRC_dev',
        'DRCD_dev',
        'afqmc-dev',
        'bustm-dev',
        'chid-dev',
        'cluewsc-dev',
        'eprstmt-dev',
        'chinese-universal-average',
        # '--- Coding ---',
        'openai_humaneval',
        'mbpp',
        'coding-average',
        # '--- Completion ---',
        'lambada',
        'story_cloze',
        'completion-average',
        # '--- EnglishUniversal ---',
        'AX_b',
        'AX_g',
        'BoolQ',
        'CB',
        'COPA',
        'MultiRC',
        'RTE',
        'ReCoRD',
        'WiC',
        'WSC',
        'english-universal-average',
        # '--- Race ---',
        'race-high',
        'race-middle',
        'race-average',
        # '--- Reasoning ---',
        'math',
        'gsm8k',
        'summedits',
        'reasoning-average',
        # '--- common QA ---',
        'hellaswag',
        'piqa',
        'winogrande',
        'openbookqa',
        'common_qa-average',
        # '--- fact QA ---',
        'nq',
        'triviaqa',
        'fact_qa-average',
        # '--- Security ---',
        'crows_pairs',
        'mmlu',
        "agieval",
        "ceval",
        "bbh",
        # "tydiqa-goldp",
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
