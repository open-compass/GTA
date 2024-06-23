_longeval_2k = ['lines_2k', 'stackselect_2k', 'textsort_2k', 'trec_e_2k']
_longeval_4k = ['2wikimqa_e_4k', 'gov_report_4k', 'lines_4k', 'stackselect_4k', 'multifieldqa_zh_4k', 'textsort_4k', 'trec_e_4k']
_longeval_8k = ['2wikimqa_e_8k', 'gov_report_8k', 'lines_8k', 'passage_retrieval_zh_8k', 'stackselect_8k', 'textsort_8k', 'trec_e_8k']
_longeval_15k = ['2wikimqa_e_15k', 'hotpotqa_e_15k', 'lines_15k', 'stackselect_15k', 'textsort_15k', 'trec_e_15k'] 
_longeval_30k = ['lines_30k', 'stackselect_30k', 'textsort_30k']

_longeval_all = _longeval_2k + _longeval_4k + _longeval_8k + _longeval_15k + _longeval_30k

longeval_summary_groups = [
    {'name': 'longeval', 'subsets': _longeval_all},
    {'name': 'longeval_2k', 'subsets': _longeval_2k},
    {'name': 'longeval_4k', 'subsets': _longeval_4k},
    {'name': 'longeval_8k', 'subsets': _longeval_8k},
    {'name': 'longeval_15k', 'subsets': _longeval_15k},
    {'name': 'longeval_30k', 'subsets': _longeval_30k},
]
