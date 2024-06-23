from opencompass.datasets import HFDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.language_quality.language_quality import LanguageExpressionEvaluator

rule_following_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="label")

rule_following_base_infer_cfg = dict(
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

rule_following_chat_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "请按规律续写(Please follow the pattern to complete the text): {prompt}"
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50),
)

rule_following_eval_cfg = dict(
    evaluator=dict(type=LanguageExpressionEvaluator),
    pred_role="BOT",
)

rule_following_base_datasets = [
    dict(
        abbr="rule-following",
        type=HFDataset,
        path='json',
        data_files="./data/language_quality/expression/rule_following.json",
        split='train',
        reader_cfg=rule_following_reader_cfg,
        infer_cfg=rule_following_base_infer_cfg,
        eval_cfg=rule_following_eval_cfg,
    )
]
rule_following_chat_datasets = [
    dict(
        abbr="rule-following",
        type=HFDataset,
        path='json',
        data_files="./data/language_quality/expression/rule_following.json",
        split='train',
        reader_cfg=rule_following_reader_cfg,
        infer_cfg=rule_following_chat_infer_cfg,
        eval_cfg=rule_following_eval_cfg,
    )
]
