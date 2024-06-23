from mmengine.config import read_base

with read_base():
    from .groups.longbench import longbench_summary_groups
    from .groups.leval import leval_summary_groups
    from .groups.longeval import longeval_summary_groups
    
summarizer = dict(
    dataset_abbrs=[
        ['leval', 'naive_average'],
        ['longbench', 'naive_average'],
        ['longeval', 'naive_average'],
        # '--------- LEval Exact Match (Acc) ---------', # category
        "LEval_coursera",
        'LEval_gsm100',
        'LEval_quality',
        "LEval_tpo",
        'LEval_topic_retrieval',
        # '--------- LEval Gen (ROUGE) ---------', # category
        'LEval_financialqa',
        'LEval_gov_report_summ',
        'LEval_legal_contract_qa',
        'LEval_meeting_summ',
        'LEval_multidocqa',
        'LEval_narrativeqa',
        'LEval_nq',
        'LEval_news_summ',
        'LEval_paper_assistant',
        'LEval_patent_summ',
        'LEval_review_summ',
        'LEval_scientificqa',
        'LEval_tvshow_summ',
        # '--------- LongBench Single-Document QA ---------', # category
        "LongBench_narrativeqa",
        'LongBench_qasper',
        'LongBench_multifieldqa_en',
        "LongBench_multifieldqa_zh",
        # '--------- LongBench Multi-Document QA ---------', # category
        'LongBench_hotpotqa',
        'LongBench_2wikimqa',
        'LongBench_musique',
        'LongBench_dureader',
        # '--------- LongBench Summarization ---------', # category
        'LongBench_gov_report',
        'LongBench_qmsum',
        'LongBench_multi_news',
        'LongBench_vcsum',
        # '--------- LongBench Few-shot Learning ---------', # category
        'LongBench_trec',
        'LongBench_triviaqa',
        'LongBench_samsum',
        'LongBench_lsht',
        # '--------- LongBench Synthetic Tasks ---------', # category
        'LongBench_passage_count',
        'LongBench_passage_retrieval_en',
        'LongBench_passage_retrieval_zh',
        # '--------- LongBench Code Completion ---------', # category
        'LongBench_lcc',
        'LongBench_repobench-p',
        # 
        ['longeval_2k', 'naive_average'],
        ['longeval_4k', 'naive_average'],
        ['longeval_8k', 'naive_average'],
        ['longeval_15k', 'naive_average'],
        ['longeval_30k', 'naive_average'],
        ['2wikimqa_e_4k', 'score'],
        ['2wikimqa_e_8k', 'score'],
        ['2wikimqa_e_15k', 'score'],
        ['gov_report_4k', 'score'],
        ['gov_report_8k', 'score'],
        ['hotpotqa_e_15k', 'score'],
        ['lines_2k', 'score'],
        ['lines_4k', 'score'],
        ['lines_8k', 'score'],
        ['lines_15k', 'score'],
        ['lines_30k', 'score'],
        ['multifieldqa_zh_4k', 'score'],
        ['passage_retrieval_zh_8k', 'score'],
        ['stackselect_2k', 'score'],
        ['stackselect_4k', 'score'],
        ['stackselect_8k', 'score'],
        ['stackselect_15k', 'score'],
        ['stackselect_30k', 'score'],
        ['textsort_2k', 'score'],
        ['textsort_4k', 'score'],
        ['textsort_8k', 'score'],
        ['textsort_15k', 'score'],
        ['textsort_30k', 'score'],
        ['trec_e_2k', 'score'],
        ['trec_e_4k', 'score'],
        ['trec_e_8k', 'score'],
        ['trec_e_15k', 'score'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []),
)
