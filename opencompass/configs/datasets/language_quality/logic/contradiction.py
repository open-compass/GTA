from opencompass.datasets.language_quality.language_quality import ContradictionNLIEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator

contradiction_reader_cfg = dict(
    input_columns=["prompt"],output_column= "reference"
)

contradiction_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=100),
)

contradiction_eval_cfg = dict(evaluator=dict(type=ContradictionNLIEvaluator),
                    # pred_postprocessor=dict(type=contradiction_postprocess)
                              )
contradiction_datasets = []
contradiction_datasets=[
    dict(
        abbr="contradiction_en",
        type=JsonlDataset,
        path='./data/language_quality/logic/contradiction_en.jsonl',
        reader_cfg=contradiction_reader_cfg,
        infer_cfg=contradiction_infer_cfg,
        eval_cfg=contradiction_eval_cfg,
    ),
    dict(
        abbr="contradiction_cn",
        type=JsonlDataset,
        path='./data/language_quality/logic/contradiction_cn.jsonl',
        reader_cfg=contradiction_reader_cfg,
        infer_cfg=contradiction_infer_cfg,
        eval_cfg=contradiction_eval_cfg,
    ),
]

