from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answer'
)

commonsenseqa_infer_cfg_zh = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="问题：{question}\n选项：\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\n答案："
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

commonsenseqa_infer_cfg_en = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="Question: {question}\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer:"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

commonsenseqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCDE'),
)

commonsenseqa_gen_datasets = [
    dict(
        type=JsonlDataset,
        abbr='commonsenseqa-zh',
        path='data/language_quality/commonsense/commonsenseqa-zh.jsonl',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg_zh,
        eval_cfg=commonsenseqa_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='commonsenseqa-en',
        path='data/language_quality/commonsense/commonsenseqa-en.jsonl',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg_en,
        eval_cfg=commonsenseqa_eval_cfg)
]
