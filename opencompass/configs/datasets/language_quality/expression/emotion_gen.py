from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.language_quality.language_quality import EmotionClsEvaluator
from opencompass.datasets import JsonlDataset

emotion_reader_cfg = dict(
    input_columns="text",
    output_column='prompt'
)

emotion_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{text}'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

emotion_eval_cfg = dict(
    evaluator=dict(type=EmotionClsEvaluator),
    pred_role="BOT",
)

emotion_datasets = [
    dict(
        abbr="emotion-all",
        type=JsonlDataset,
        path="./data/language_quality/expression/emotion.jsonl",
        reader_cfg=emotion_reader_cfg,
        infer_cfg=emotion_infer_cfg,
        eval_cfg=emotion_eval_cfg,
    ),
]
