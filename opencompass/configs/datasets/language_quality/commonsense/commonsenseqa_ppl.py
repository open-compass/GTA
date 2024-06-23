from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import JsonlDataset

commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answer'
)

commonsenseqa_infer_cfg_zh = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': "{question}\n回答：{A}",
            'B': "{question}\n回答：{B}",
            'C': "{question}\n回答：{C}",
            'D': "{question}\n回答：{D}",
            'E': "{question}\n回答：{E}",
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

commonsenseqa_infer_cfg_en = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': "{question}\nAnswer: {A}",
            'B': "{question}\nAnswer: {B}",
            'C': "{question}\nAnswer: {C}",
            'D': "{question}\nAnswer: {D}",
            'E': "{question}\nAnswer: {E}",
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

commonsenseqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

commonsenseqa_ppl_datasets = [
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
