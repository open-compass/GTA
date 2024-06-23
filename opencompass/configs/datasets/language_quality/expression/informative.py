from opencompass.datasets import HFDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.language_quality.language_quality import LanguageExpressionEvaluator

informative_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="label")

informative_base_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "{prompt}"
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=100),
)

informative_chat_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "请续写(Please complete the text): {prompt}"
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=100),
)

informative_eval_cfg = dict(
    evaluator=dict(type=LanguageExpressionEvaluator),
    pred_role="BOT",
)

informative_base_datasets = [
    dict(
        abbr="informative",
        type=HFDataset,
        path='json',
        data_files="./data/language_quality/expression/informative.json",
        split='train',
        reader_cfg=informative_reader_cfg,
        infer_cfg=informative_base_infer_cfg,
        eval_cfg=informative_eval_cfg,
    )
]
informative_chat_datasets = [
    dict(
        abbr="informative",
        type=HFDataset,
        path='json',
        data_files="./data/language_quality/expression/informative.json",
        split='train',
        reader_cfg=informative_reader_cfg,
        infer_cfg=informative_chat_infer_cfg,
        eval_cfg=informative_eval_cfg,
    )
]
