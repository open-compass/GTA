from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.mmlu import mmlu_summary_groups
    from ...summarizers.groups.ceval import ceval_summary_groups
    from ...summarizers.groups.bbh import bbh_summary_groups
    from ...summarizers.groups.GaokaoBench import GaokaoBench_summary_groups
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # '--------- 知识 Knowledge ---------',
        ['mmlu', 'naive_average'],
        ['cmmlu', 'naive_average'],
        ['ceval-test', 'naive_average'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa', 'score'],
        # ['triviaqa_wiki_1shot', 'score'],
        ['nq', 'score'],
        # ['nq_open_1shot', 'score'],
        # '--------- 语言 Language ---------',
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
        ['cluewsc-dev', 'accuracy'],  # 为了看指令跟随
        ['csl_dev', 'accuracy'],  # 为了看指令跟随
        # '--------- 推理 Reasoning ---------',
        ['hellaswag', 'accuracy'],
        ['bbh', 'naive_average'],
        ['TheoremQA', 'accuracy'],
        # '--------- 数学 Mathematics ---------',
        ['gsm8k', 'accuracy'],
        ['math', 'accuracy'],
        # '--------- 代码 Coding ---------',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['mbpp', 'score'],
        # ['sanitized_mbpp', 'score'],
        # '--------- 数学智能体 Math Agent ---------',
        ['math-agent', 'follow_acc'],
        ['math-agent', 'reasoning_acc'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []),
)
