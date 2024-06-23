from mmengine.config import read_base
from opencompass.summarizers import SubjectiveSummarizer, PretrainSummarizer

longeval_2k_summary_groups = [
    {
        "name": "longeval_2k",
        "subsets": [
            "lines_2k",
            "stackselect_2k",
            "textsort_2k",
            "trec_e_2k",
        ]
    }
]

longeval_4k_summary_groups = [
    {
        "name": "longeval_4k",
        "subsets": [
            "2wikimqa_e_4k",
            "gov_report_4k",
            "multifieldqa_zh_4k",
            "lines_4k",
            "stackselect_4k",
            "textsort_4k",
            "trec_e_4k",
        ]
    }
]

longeval_8k_summary_groups = [
    {
        "name": "longeval_8k",
        "subsets": [
            "2wikimqa_e_8k",
            "gov_report_8k",
            "lines_8k",
            "passage_retrieval_zh_8k",
            "stackselect_8k",
            "textsort_8k",
            "trec_e_8k",
        ]
    }
]

longeval_15k_summary_groups = [
    {
        "name": "longeval_15k",
        "subsets": [
            "2wikimqa_e_15k",
            "hotpotqa_e_15k",
            "lines_15k",
            "stackselect_15k",
            "textsort_15k",
            "trec_e_15k",
        ]
    }
]

longeval_30k_summary_groups = [
    {
        "name": "longeval_30k",
        "subsets": [
            "lines_30k",
            "stackselect_30k",
            "textsort_30k"
        ]
    }
]

summarizer = dict(
    type=PretrainSummarizer,
    dataset_abbrs=[
        "longeval_2k",
        "longeval_4k",
        "longeval_8k",
        "longeval_15k",
        "longeval_30k",

        "lines_2k",
        "stackselect_2k",
        "textsort_2k",
        "trec_e_2k",

        "2wikimqa_e_4k",
        "gov_report_4k",
        "multifieldqa_zh_4k",
        "lines_4k",
        "stackselect_4k",
        "textsort_4k",
        "trec_e_4k",

        "2wikimqa_e_8k",
        "gov_report_8k",
        "lines_8k",
        "passage_retrieval_zh_8k",
        "stackselect_8k",
        "textsort_8k",
        "trec_e_8k",

        "2wikimqa_e_15k",
        "hotpotqa_e_15k",
        "lines_15k",
        "stackselect_15k",
        "textsort_15k",
        "trec_e_15k",

        "lines_30k",
        "stackselect_30k",
        "textsort_30k",
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
