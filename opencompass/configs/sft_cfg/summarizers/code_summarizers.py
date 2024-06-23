from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.ds1000 import ds1000_summary_groups


code_passk_summary_groups = [
    # rename
    {'name': 'humaneval_pass@1(greedy)', 'subsets': [['openai_humaneval', 'humaneval_pass@1']]},
    {'name': 'humaneval_pass@10', 'subsets': [['openai_humaneval_repeat10', 'humaneval_pass@10']]},
    {'name': 'humaneval_cn_pass@1(greedy)', 'subsets': [['openai_humaneval_cn', 'humaneval_pass@1']]},
    {'name': 'humaneval_cn_pass@10', 'subsets': [['openai_humaneval_cn_repeat10', 'humaneval_pass@10']]},
    {'name': 'humaneval_plus_pass@1(greedy)', 'subsets': [['humaneval_plus', 'humaneval_plus_pass@1']]},
    {'name': 'humaneval_plus_pass@10', 'subsets': [['humaneval_plus_repeat10', 'humaneval_plus_pass@10']]},
    {'name': 'mbpp_pass@1(greedy)', 'subsets': [['mbpp', 'score']]},
    {'name': 'mbpp_pass@10', 'subsets': [['mbpp_repeat10', 'pass@10']]},
    {'name': 'mbpp_cn_pass@1(greedy)', 'subsets': [['mbpp_cn', 'score']]},
    {'name': 'mbpp_cn_pass@10', 'subsets': [['mbpp_cn_repeat10', 'pass@10']]},
    {'name': 'sanitized_mbpp_pass@1(greedy)', 'subsets': [['sanitized_mbpp', 'score']]},
    {'name': 'sanitized_mbpp_pass@10', 'subsets': [['sanitized_mbpp_repeat10', 'pass@10']]},
    
    {'name': 'ds1000_pass@1(greedy)', 'subsets': [['ds1000', 'naive_average']]},
    {'name': 'py150_pass@1(greedy)', 'subsets': [['py150', 'score']]},
    {'name': 'maxmin_pass@1(greedy)', 'subsets': [['maxmin', 'accuracy']]},
    {'name': 'mbpp_plus_pass@1(greedy)', 'subsets': [['mbpp_plus', 'mbpp_plus_pass@1']]},
    # real add
    {'name': 'humanevalx', 'subsets': ['humanevalx-python', 'humanevalx-cpp', 'humanevalx-go', 'humanevalx-java', 'humanevalx-js']},
    # {'name': 'code', 'subsets': ['humaneval_plus_pass@1(greedy)', 'sanitized_mbpp_pass@1(greedy)', 'humanevalx', '']}
]

summarizer = dict(
    dataset_abbrs=[
        # 'code',
        'humaneval_pass@1(greedy)',
        'humaneval_pass@10',
        'humaneval_plus_pass@1(greedy)',
        'humaneval_plus_pass@10',

        'mbpp_pass@1(greedy)',
        'mbpp_pass@10',
        'sanitized_mbpp_pass@1(greedy)',
        'sanitized_mbpp_pass@10',
        'mbpp_plus_pass@1(greedy)',

        'py150_pass@1(greedy)',
        'maxmin_pass@1(greedy)',

        'humanevalx',
        'humanevalx-python',
        'humanevalx-cpp',
        'humanevalx-go',
        'humanevalx-java',
        'humanevalx-js',

        'ds1000_pass@1(greedy)',
        ['ds1000_Pandas', 'accuracy'],
        ['ds1000_Numpy', 'accuracy'],
        ['ds1000_Tensorflow', 'accuracy'],
        ['ds1000_Scipy', 'accuracy'],
        ['ds1000_Sklearn', 'accuracy'],
        ['ds1000_Pytorch', 'accuracy'],
        ['ds1000_Matplotlib','accuracy'],

        'humaneval_cn_pass@1(greedy)',
        'humaneval_cn_pass@10',
        'mbpp_cn_pass@1(greedy)',
        'mbpp_cn_pass@10',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
