from opencompass.summarizers.summarizer_pretrain import PretrainSummarizer

longbench_no_code = [
    'LongBench_2wikimqa',
    'LongBench_hotpotqa',
    'LongBench_musique',
    'LongBench_multifieldqa_en',
    "LongBench_multifieldqa_zh",
    "LongBench_narrativeqa",
    'LongBench_qasper',
    'LongBench_triviaqa',
    'LongBench_gov_report',
    'LongBench_qmsum',
    'LongBench_vcsum',
    'LongBench_dureader',
    'LongBench_passage_retrieval_en',
    'LongBench_passage_retrieval_zh',
    'LongBench_passage_count',
    'LongBench_trec',
    'LongBench_lsht',
    'LongBench_multi_news',
    'LongBench_samsum',
]

longbench_code = [
    'LongBench_lcc',
    'LongBench_repobench-p',
]

longbench_summary_groups = [
    {'name': 'average_except_code', 'subsets': longbench_no_code},
    {'name': 'average_code', 'subsets': longbench_code}
]

summarizer = dict(
    type=PretrainSummarizer,
    dataset_abbrs = [
        'LongBench_lcc',
        'LongBench_repobench-p',  
      
        'average_code',

        'LongBench_hotpotqa',
        'LongBench_2wikimqa',
        'LongBench_musique',
        'LongBench_dureader',

        'LongBench_multifieldqa_en',
        "LongBench_multifieldqa_zh",
        "LongBench_narrativeqa",
        'LongBench_qasper',

        'LongBench_gov_report',
        'LongBench_qmsum',
        'LongBench_multi_news',
        'LongBench_vcsum',

        'LongBench_triviaqa',
        'LongBench_samsum',
        'LongBench_trec',
        'LongBench_lsht',

        'LongBench_passage_retrieval_en',
        'LongBench_passage_retrieval_zh',
        'LongBench_passage_count',

        'average_except_code',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
