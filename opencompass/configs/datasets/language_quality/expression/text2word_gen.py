from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import RandomRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.language_quality.language_quality import Text2WordEvaluator
from opencompass.datasets import JsonlDataset

text2word_reader_cfg = dict(
    input_columns=['explanation'],
    output_column='word')

text2word_infer_cfg_zh = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "描述：{explanation}\n词：",
                ),
                dict(
                    role="BOT",
                    prompt="{word}",
                )]),
        ice_token="</E>",
    ),
    retriever=dict(type=RandomRetriever, ice_num=5),
    inferencer=dict(type=GenInferencer)
)

text2word_infer_cfg_en = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "Description: {explanation}\nWord: ",
                ),
                dict(
                    role="BOT",
                    prompt="{word}",
                )]),
        ice_token="</E>",
    ),
    retriever=dict(type=RandomRetriever, ice_num=5),
    inferencer=dict(type=GenInferencer)
)

text2word_eval_cfg = dict(
    evaluator=dict(
        type=Text2WordEvaluator,
    ),
    pred_role="BOT",
)

text2word_datasets = [
    dict(
        type=JsonlDataset,
        abbr='text2word-zh-v2',
        path='data/language_quality/expression/text2word-zh.jsonl',
        reader_cfg=text2word_reader_cfg,
        infer_cfg=text2word_infer_cfg_zh,
        eval_cfg=text2word_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='text2word-en-v1',
        path='data/language_quality/expression/text2word-en.jsonl',
        reader_cfg=text2word_reader_cfg,
        infer_cfg=text2word_infer_cfg_en,
        eval_cfg=text2word_eval_cfg),
]
