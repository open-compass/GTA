from opencompass.openicl import GenInferencer
from opencompass.datasets import JsonlDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.language_quality.language_quality import winogradEvaluator

winograd_cn_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='label',
    train_split="train",
    test_split="train")

winograd_cn_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}’{pronoun}‘指A.{opt1},B.{opt2},请回答A或者B：",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,max_out_len=10))

winograd_cn_eval_cfg = dict(evaluator=dict(type=winogradEvaluator), )

winograd_cn_api_datasets = [
    dict(
        abbr='winograd_cn',
        type=JsonlDataset,
        path='./data/language_quality/logic/wingrad_zh.jsonl',
        reader_cfg=winograd_cn_reader_cfg,
        infer_cfg=winograd_cn_infer_cfg,
        eval_cfg=winograd_cn_eval_cfg)
]
