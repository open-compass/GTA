ds1000_summary_groups = []

# ds1000
_ds1000 = [
    'Pandas',
    'Numpy',
    'Tensorflow',
    'Scipy',
    'Sklearn',
    'Pytorch',
    'Matplotlib',
]
_ds1000 = ['ds1000_' + s for s in _ds1000]
ds1000_summary_groups.append({'name': 'ds1000', 'subsets': _ds1000})
