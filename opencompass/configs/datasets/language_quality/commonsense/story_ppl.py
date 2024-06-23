from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import JsonlDataset

story_reader_cfg = dict(
    input_columns=['story', 'choice1', 'choice2'],
    output_column='answer')

story_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            1: "{story} {choice1}",
            2: "{story} {choice2}",
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

story_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

story_ppl_datasets = [
    dict(
        type=JsonlDataset,
        abbr='story-zh',
        path='data/language_quality/commonsense/story-zh.jsonl',
        reader_cfg=story_reader_cfg,
        infer_cfg=story_infer_cfg,
        eval_cfg=story_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='story-en',
        path='data/language_quality/commonsense/story-en.jsonl',
        reader_cfg=story_reader_cfg,
        infer_cfg=story_infer_cfg,
        eval_cfg=story_eval_cfg)
]
