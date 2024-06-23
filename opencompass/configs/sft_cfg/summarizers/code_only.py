from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.ds1000 import ds1000_summary_groups

summarizer = dict(
    dataset_abbrs = [
        # '代码', # subcategory
        'openai_humaneval',
        'mbpp',
        'maxmin',
        'py150',
        'ds1000',
        'ds1000_Pandas', 
        'ds1000_Numpy', 
        'ds1000_Tensorflow', 
        'ds1000_Scipy', 
        'ds1000_Sklearn', 
        'ds1000_Pytorch', 
        'ds1000_Matplotlib'
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
