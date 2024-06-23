from mmengine.config import read_base

with read_base():
    from ..groups.GaokaoBench import GaokaoBench_summary_groups

summarizer = dict(
    dataset_abbrs = [
        "GaokaoBench",
        "GaokaoBench_2010-2022_Chinese_Lang_and_Usage_MCQs",
        "GaokaoBench_2010-2022_Chinese_Modern_Lit",
        "GaokaoBench_2010-2022_Math_I_MCQs",
        "GaokaoBench_2010-2022_Math_II_MCQs",
        "GaokaoBench_2010-2013_English_MCQs",
        "GaokaoBench_2010-2022_English_Fill_in_Blanks",
        "GaokaoBench_2012-2022_English_Cloze_Test",
        "GaokaoBench_2010-2022_English_Reading_Comp",
        "GaokaoBench_2010-2022_Physics_MCQs",
        "GaokaoBench_2010-2022_Chemistry_MCQs",
        "GaokaoBench_2010-2022_Biology_MCQs",
        "GaokaoBench_2010-2022_History_MCQs",
        "GaokaoBench_2010-2022_Political_Science_MCQs",
        "GaokaoBench_2010-2022_Geography_MCQs",
        # "GaokaoBench_2010-2022_Math_I_Fill-in-the-Blank",
        # "GaokaoBench_2010-2022_Math_II_Fill-in-the-Blank",
        # "GaokaoBench_2010-2022_Chinese_Language_Famous_Passages_and_Sentences_Dictation",
        # "GaokaoBench_2014-2022_English_Language_Cloze_Passage",
        # "GaokaoBench_2010-2022_Geography_Open-ended_Questions",
        # "GaokaoBench_2010-2022_Chemistry_Open-ended_Questions",
        # "GaokaoBench_2010-2022_Math_I_Open-ended_Questions",
        # "GaokaoBench_2010-2022_History_Open-ended_Questions",
        # "GaokaoBench_2010-2022_Biology_Open-ended_Questions",
        # "GaokaoBench_2010-2022_Math_II_Open-ended_Questions",
        # "GaokaoBench_2010-2022_Physics_Open-ended_Questions",
        # "GaokaoBench_2010-2022_Political_Science_Open-ended_Questions",
        # "GaokaoBench_2012-2022_English_Language_Error_Correction",
        # "GaokaoBench_2010-2022_Chinese_Language_Ancient_Poetry_Reading",
        # "GaokaoBench_2010-2022_Chinese_Language_Practical_Text_Reading",
        # "GaokaoBench_2010-2022_Chinese_Language_Literary_Text_Reading",
        # "GaokaoBench_2010-2022_Chinese_Language_Classical_Chinese_Reading",
        # "GaokaoBench_2010-2022_Chinese_Language_Language_and_Writing_Skills_Open-ended_Questions",
    ],
    summary_groups=GaokaoBench_summary_groups,
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore')
)
