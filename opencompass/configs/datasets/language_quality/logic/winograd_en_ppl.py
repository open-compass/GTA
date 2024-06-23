from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.datasets import JsonlDataset
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.language_quality.language_quality import winogradCNDataset, winogradEvaluator

winograd_en_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='label',
    train_split="train",
    test_split="train",
    test_range="[:100]"
)

winograd_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            "{prompt}'{pronoun}'refer to {opt1}.",  # noqa
            1:
            "{prompt}'{pronoun}'refer to {opt2}.",  # noqa
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

winograd_en_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

winograd_en_ppl_datasets = [
    dict(
        abbr='winograd_en',
        type=JsonlDataset,
        path='./data/language_quality/logic/wingrad.jsonl',
        reader_cfg=winograd_en_reader_cfg,
        infer_cfg=winograd_en_infer_cfg,
        eval_cfg=winograd_en_eval_cfg)
]
