from mmengine.config import read_base

with read_base():
    from ..groups.agieval import agieval_summary_groups

summarizer = dict(
    dataset_abbrs = [
        # AGIEval
        'agieval-aqua-rat',
        'agieval-math',
        'agieval-logiqa-en',
        'agieval-logiqa-zh',
        'agieval-jec-qa-kd',
        'agieval-jec-qa-ca',
        'agieval-lsat-ar',
        'agieval-lsat-lr',
        'agieval-lsat-rc',
        'agieval-sat-math',
        'agieval-sat-en',
        'agieval-sat-en-without-passage',
        'agieval-gaokao-chinese',
        'agieval-gaokao-english',
        'agieval-gaokao-geography',
        'agieval-gaokao-history',
        'agieval-gaokao-biology',
        'agieval-gaokao-chemistry',
        'agieval-gaokao-physics',
        'agieval-gaokao-mathqa',
        'agieval-gaokao-mathcloze',
        'agieval-gaokao',
        'agieval-gaokao-full',
        'agieval',
    ],
    summary_groups=agieval_summary_groups,
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore')
)
