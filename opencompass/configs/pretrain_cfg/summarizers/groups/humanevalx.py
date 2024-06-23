humanevalx_summary_groups = []

_humanevalx = [
    'go',
    'cpp',
    'java',
    'js',
    'python',
    # 'rust',
]
_humanevalx = ['humanevalx-' + s for s in _humanevalx]
humanevalx_summary_groups.append({'name': 'humanevalx', 'subsets': _humanevalx})
