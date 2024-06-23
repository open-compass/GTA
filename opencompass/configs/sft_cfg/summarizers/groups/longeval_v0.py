# deprecated move to ./longeval.py
# 这个 group 是按照数据集划分的，新版本是按照 length 划分

_2wikimqa_e = ['2wikimqa_e_4k', '2wikimqa_e_8k', '2wikimqa_e_15k']
_gov_report = ['gov_report_4k', 'gov_report_8k']
_hotpotqa_e = ['hotpotqa_e_15k']
_lines = ['lines_2k', 'lines_4k', 'lines_8k', 'lines_15k', 'lines_30k']
_multifieldqa_zh = ['multifieldqa_zh_4k']
_passage_retrieval_zh = ['passage_retrieval_zh_8k']
_stackselect = ['stackselect_2k', 'stackselect_4k', 'stackselect_8k', 'stackselect_15k', 'stackselect_30k']
_textsort = ['textsort_2k', 'textsort_4k', 'textsort_8k', 'textsort_15k', 'textsort_30k']
_trec_e = ['trec_e_2k', 'trec_e_4k', 'trec_e_8k', 'trec_e_15k']

_longeval_all = _2wikimqa_e + _gov_report + _hotpotqa_e + _lines + _multifieldqa_zh + _passage_retrieval_zh + _stackselect + _textsort + _trec_e

longeval_summary_groups = [
    {'name': 'longeval', 'subsets': _longeval_all},
    {'name': '2wikimqa_e', 'subsets': _2wikimqa_e},
    {'name': 'gov_report', 'subsets': _gov_report},
    {'name': 'lines', 'subsets': _lines},
    {'name': 'stackselect', 'subsets': _stackselect},
    {'name': 'textsort', 'subsets': _textsort},
    {'name': 'trec_e', 'subsets': _trec_e},
]
