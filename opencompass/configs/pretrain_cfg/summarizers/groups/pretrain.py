chinese_summary_groups = []

_chinese = ['CMRC_dev',
            'DRCD_dev',
            'afqmc-dev',
            'bustm-dev',
            'chid-dev',
            'cluewsc-dev',
            'eprstmt-dev']

chinese_summary_groups.append({'name': 'chinese-universal-average', 'subsets': _chinese})

coding_summary_groups = []

_coding = ['openai_humaneval',
           'mbpp',
           'py150',
           'maxmin']

coding_summary_groups.append({'name': 'coding-average', 'subsets': _coding})

common_qa_summary_groups = []

_common_qa = ['hellaswag',
              'piqa',
              'winogrande',
              'openbookqa']

common_qa_summary_groups.append({'name': 'common_qa-average', 'subsets': _common_qa})


completion_summary_groups = []

_completion = ['lambada',
               'story_cloze']

completion_summary_groups.append({'name': 'completion-average', 'subsets': _completion})


english_summary_groups = []

_english = ['AX_b',
            'AX_g',
            'BoolQ',
            'CB',
            'COPA',
            'MultiRC',
            'RTE',
            'ReCoRD',
            'WiC',
            'WSC']

english_summary_groups.append({'name': 'english-universal-average', 'subsets': _english})


fact_qa_summary_groups = []

_fact_qa = ['nq',
            'triviaqa']

fact_qa_summary_groups.append({'name': 'fact_qa-average', 'subsets': _fact_qa})

race_summary_groups = []

_race = ['race-high',
         'race-middle']

race_summary_groups.append({'name': 'race-average', 'subsets': _race})

reasoning_summary_groups = []

_reasoning = ['math',
              'gsm8k',
              'summedits']

reasoning_summary_groups.append({'name': 'reasoning-average', 'subsets': _reasoning})


