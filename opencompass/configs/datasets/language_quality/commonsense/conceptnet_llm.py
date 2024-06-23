from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, RandomRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.datasets.language_quality.language_quality import json_postprocess

conceptnet_reader_cfg = dict(
    input_columns=['start', 'relation', 'references'],
    output_column='ends')

conceptnet_infer_cfg_zh = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "实体：{start}，关系：{relation}",
                ),
                dict(
                    role="BOT",
                    prompt="和实体构成对应关系的词：{ends}",
                )]),
        ice_token="</E>",
    ),
    retriever=dict(type=RandomRetriever, ice_num=5),
    inferencer=dict(type=GenInferencer)
)

conceptnet_infer_cfg_en = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "Entity: {start}. Relation: {relation}.",
                ),
                dict(
                    role="BOT",
                    prompt="Words that can form a corresponding relation with the entity: {ends}",
                )]),
        ice_token="</E>",
    ),
    retriever=dict(type=RandomRetriever, ice_num=5),
    inferencer=dict(type=GenInferencer)
)

conceptnet_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        prompt="你是一个常识知识评分系统，给定一个实体和一个关系，对回答进行判断。请以JSON格式返回响应，其中包含特定的键和值。\n"
                               "要求：\n1. 'has_commonsense'：对照参考词组，判断回答的词组中是否存在能与给定实体构成对应关系的词，用True和False回答。\n"
                               "2. 'has_conflict'：根据常识，判断回答的词组中是否存在与给定实体构成对应关系违背常识的词，用True和False回答。\n"
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="词：{start}，关系：{relation}\n参考词组：{references}\n回答：{prediction}")])),
        postprocessor=dict(
            type=json_postprocess
        )
    ),
    pred_role="BOT",
)

conceptnet_datasets = [
    dict(
        type=JsonlDataset,
        abbr='conceptnet-zh',
        path='data/language_quality/commonsense/conceptnet-zh.jsonl',
        reader_cfg=conceptnet_reader_cfg,
        infer_cfg=conceptnet_infer_cfg_zh,
        eval_cfg=conceptnet_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='conceptnet-en',
        path='data/language_quality/commonsense/conceptnet-en.jsonl',
        reader_cfg=conceptnet_reader_cfg,
        infer_cfg=conceptnet_infer_cfg_en,
        eval_cfg=conceptnet_eval_cfg),
]
