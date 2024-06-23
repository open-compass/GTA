from mmengine.config import read_base

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.flores import flores_summary_groups
    from .groups.jigsaw_multilingual import jigsaw_multilingual_summary_groups

summarizer = dict(
    dataset_abbrs = [
        # '--------- 考试 Exam ---------', # category
        # 'Mixed', # subcategory
        "ceval",
        'agieval',
        'mmlu',
        'mmlu_cn', # placeholder
        "GaokaoBench",
        'ARC-c',
        # '自建考试', # subcategory
        'compass_exam-senior-high-2023',
        # '--------- 语言 Language ---------', # category
        # '字词释义', # subcategory
        'WiC',
        'summedits',
        # '成语习语', # subcategory
        'chid-dev',
        # '语义相似度', # subcategory
        'afqmc-dev',
        'bustm-dev',
        # '指代消解', # subcategory
        'cluewsc-dev',
        'WSC',
        'winogrande',
        # '翻译', # subcategory
        'flores_100',
        # '--------- 知识 Knowledge ---------', # category
        # '知识问答', # subcategory
        'BoolQ',
        'commonsense_qa',
        'nq',
        'triviaqa',
        # '多语种问答', # subcategory
        'tydiqa', # placeholder
        # '--------- 推理 Reasoning ---------', # category
        # '文本蕴含', # subcategory
        'cmnli',
        'ocnli',
        'ocnli_fc-dev',
        'AX_b',
        'AX_g',
        'CB',
        'RTE',
        # '常识推理', # subcategory
        'story_cloze',
        'story_cloze_cn', # placeholder
        'COPA',
        'ReCoRD',
        'hellaswag',
        'piqa',
        'siqa',
        'strategyqa',
        # '数学推理', # subcategory
        'math',
        'math_cn', # placeholder
        'gsm8k',
        'gsm8k_cn', # placeholder
        # '定理应用', # subcategory
        'TheoremQA',
        # '代码', # subcategory
        'openai_humaneval',
        'mbpp',
        # '综合推理', # subcategory
        "bbh",
        # '--------- 理解 Understanding ---------', # category
        # '阅读理解', # subcategory
        'C3',
        'CMRC_dev',
        'DRCD_dev',
        'MultiRC',
        'race-middle',
        'race-high',
        'openbookqa_fact',
        # '内容总结', # subcategory
        'csl_dev',
        'lcsts',
        'Xsum',
        # '内容分析', # subcategory
        'eprstmt-dev',
        'lambada',
        'tnews-dev',
        # '--------- 安全 Safety ---------', # category
        # '偏见', # subcategory
        'crows_pairs',
        'crowspairs_cn', # placeholder
        'civil_comments',
        # '有害性', # subcategory
        'jigsaw_multilingual',
        "allenai_real-toxicity-prompts",
        # '真实性', # subcategory
        ('truthful_qa', 'truth'),
        ('truthful_qa', 'info'),
        # '--------- ceval 细节 ---------',
        "ceval-stem",
        "ceval-social-science",
        "ceval-humanities",
        "ceval-other",
        "ceval-hard",
        # category
        'ceval-advanced_mathematics',
        'ceval-college_chemistry',
        'ceval-college_physics',
        'ceval-college_programming',
        'ceval-computer_architecture',
        'ceval-computer_network',
        'ceval-discrete_mathematics',
        'ceval-electrical_engineer',
        'ceval-high_school_biology',
        'ceval-high_school_chemistry',
        'ceval-high_school_mathematics',
        'ceval-high_school_physics',
        'ceval-metrology_engineer',
        'ceval-middle_school_biology',
        'ceval-middle_school_chemistry',
        'ceval-middle_school_mathematics',
        'ceval-middle_school_physics',
        'ceval-operating_system',
        'ceval-probability_and_statistics',
        'ceval-veterinary_medicine',
        'ceval-business_administration',
        'ceval-college_economics',
        'ceval-education_science',
        'ceval-high_school_geography',
        'ceval-high_school_politics',
        'ceval-mao_zedong_thought',
        'ceval-marxism',
        'ceval-middle_school_geography',
        'ceval-middle_school_politics',
        'ceval-teacher_qualification',
        'ceval-art_studies',
        'ceval-chinese_language_and_literature',
        'ceval-high_school_chinese',
        'ceval-high_school_history',
        'ceval-ideological_and_moral_cultivation',
        'ceval-law',
        'ceval-legal_professional',
        'ceval-logic',
        'ceval-middle_school_history',
        'ceval-modern_chinese_history',
        'ceval-professional_tour_guide',
        'ceval-accountant',
        'ceval-basic_medicine',
        'ceval-civil_servant',
        'ceval-clinical_medicine',
        'ceval-environmental_impact_assessment_engineer',
        'ceval-fire_engineer',
        'ceval-physician',
        'ceval-plant_protection',
        'ceval-sports_science',
        'ceval-tax_accountant',
        'ceval-urban_and_rural_planner',
        # '--------- agieval 细节 ---------',
        'agieval-chinese',
        'agieval-english',
        'agieval-gaokao',
        # category
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
        # '--------- mmlu 细节 ---------',
        'mmlu-humanities',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-other',
        # category
        'lukaemon_mmlu_abstract_algebra',
        'lukaemon_mmlu_anatomy',
        'lukaemon_mmlu_astronomy',
        'lukaemon_mmlu_business_ethics',
        'lukaemon_mmlu_clinical_knowledge',
        'lukaemon_mmlu_college_biology',
        'lukaemon_mmlu_college_chemistry',
        'lukaemon_mmlu_college_computer_science',
        'lukaemon_mmlu_college_mathematics',
        'lukaemon_mmlu_college_medicine',
        'lukaemon_mmlu_college_physics',
        'lukaemon_mmlu_computer_security',
        'lukaemon_mmlu_conceptual_physics',
        'lukaemon_mmlu_econometrics',
        'lukaemon_mmlu_electrical_engineering',
        'lukaemon_mmlu_elementary_mathematics',
        'lukaemon_mmlu_formal_logic',
        'lukaemon_mmlu_global_facts',
        'lukaemon_mmlu_high_school_biology',
        'lukaemon_mmlu_high_school_chemistry',
        'lukaemon_mmlu_high_school_computer_science',
        'lukaemon_mmlu_high_school_european_history',
        'lukaemon_mmlu_high_school_geography',
        'lukaemon_mmlu_high_school_government_and_politics',
        'lukaemon_mmlu_high_school_macroeconomics',
        'lukaemon_mmlu_high_school_mathematics',
        'lukaemon_mmlu_high_school_microeconomics',
        'lukaemon_mmlu_high_school_physics',
        'lukaemon_mmlu_high_school_psychology',
        'lukaemon_mmlu_high_school_statistics',
        'lukaemon_mmlu_high_school_us_history',
        'lukaemon_mmlu_high_school_world_history',
        'lukaemon_mmlu_human_aging',
        'lukaemon_mmlu_human_sexuality',
        'lukaemon_mmlu_international_law',
        'lukaemon_mmlu_jurisprudence',
        'lukaemon_mmlu_logical_fallacies',
        'lukaemon_mmlu_machine_learning',
        'lukaemon_mmlu_management',
        'lukaemon_mmlu_marketing',
        'lukaemon_mmlu_medical_genetics',
        'lukaemon_mmlu_miscellaneous',
        'lukaemon_mmlu_moral_disputes',
        'lukaemon_mmlu_moral_scenarios',
        'lukaemon_mmlu_nutrition',
        'lukaemon_mmlu_philosophy',
        'lukaemon_mmlu_prehistory',
        'lukaemon_mmlu_professional_accounting',
        'lukaemon_mmlu_professional_law',
        'lukaemon_mmlu_professional_medicine',
        'lukaemon_mmlu_professional_psychology',
        'lukaemon_mmlu_public_relations',
        'lukaemon_mmlu_security_studies',
        'lukaemon_mmlu_sociology',
        'lukaemon_mmlu_us_foreign_policy',
        'lukaemon_mmlu_virology',
        'lukaemon_mmlu_world_religions',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
