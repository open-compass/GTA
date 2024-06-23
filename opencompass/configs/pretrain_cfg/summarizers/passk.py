from opencompass.summarizers import PretrainSummarizer

n = 30
k = [5, 10, 100]

dataset_abbrs = []

for _k in k:
    dataset_abbrs.extend([
        [f'openai_humaneval_repeat{n}', f'humaneval_pass@{_k}'],
        [f'mbpp_repeat{n}', f'pass@{_k}']
    ])

summarizer = dict(
    type=PretrainSummarizer,
    dataset_abbrs=dataset_abbrs,
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
