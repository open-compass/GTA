from opencompass.datasets import JsonlDataset
from opencompass.datasets.language_quality.language_quality import coreference_resolution_postprocess, CREvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator

coreference_resolution_reader_cfg = dict(
    input_columns=["prompt"],output_column= "answer"
)
shot_1 = "奖杯不适合放在棕色手提箱里，因为它太大了。它指（奖杯）。"
shot_2 = "保罗试图通过电话给乔治打电话，但他不在。他指（乔治）。"
shot_3 = "律师问了证人一个问题，但他（证人）不愿意回答。他指（证人）。"
shot_4 = "安娜在考试中的表现比她的好朋友露西差得多，因为她学习太努力了。她指（露西）。"
shot_5 = "The trophy is not suitable to be put in the brown suitcase because it's too big. It refers to the (trophy)."
shot_6 = "Paul tried to call George over the phone, but he was not there. He refers to (George)."
shot_7 = "The lawyer asked the witness a question, but he (the witness) didn't want to answer. He refers to (the witness)."
shot_8 = "Anna performed much worse in the exam than her good friend Lucy, because she studied too hard. She refers to (Lucy)."
coreference_resolution_cn_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=f"{shot_1}{shot_2}{shot_3}{shot_4}{{prompt}}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=10),
)
coreference_resolution_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=f"{shot_5}{shot_6}{shot_7}{shot_8}{{prompt}}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=10),
)

coreference_resolution_eval_cfg = dict(evaluator=dict(type=CREvaluator),
                    pred_postprocessor=dict(type=coreference_resolution_postprocess))

coreference_resolution_datasets=[
    dict(
        abbr="coreference_resolution_en",
        type=JsonlDataset,
        path='./data/language_quality/logic/coreference_resolution_en.jsonl',
        reader_cfg=coreference_resolution_reader_cfg,
        infer_cfg=coreference_resolution_en_infer_cfg,
        eval_cfg=coreference_resolution_eval_cfg,
    ),
    dict(
        abbr="coreference_resolution_cn",
        type=JsonlDataset,
        path='./data/language_quality/logic/coreference_resolution_cn.jsonl',
        reader_cfg=coreference_resolution_reader_cfg,
        infer_cfg=coreference_resolution_cn_infer_cfg,
        eval_cfg=coreference_resolution_eval_cfg,
    )
]

