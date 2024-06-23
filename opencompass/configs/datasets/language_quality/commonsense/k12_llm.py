from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.datasets.language_quality.language_quality import first_number_postprocess_plus

k12_reader_cfg = dict(
    input_columns=['story', 'question'],
    output_column='category')

k12_infer_cfg_zh = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='问题：{question}\n回答：'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

k12_infer_cfg_en = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='Question: {question}\nAnswer:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

k12_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        prompt="请判断下列问题的回答是否合理，可参考给定故事。对回答进行0~100的打分，分数越高表示回答越准确。"
                               "如果回答的语言和问题的语言不一致（翻译问题除外），直接打分为0分。"
                               "如果回答没有正面回答问题内容，直接打分为0分。"
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="问题：{question}\n给定故事：{story}\n回答：{prediction}")])),
        postprocessor=dict(
            type=first_number_postprocess_plus
        )
    ),
    pred_role="BOT",
)

k12_datasets = [
    dict(
        type=JsonlDataset,
        abbr='k12-zh',
        path='data/language_quality/commonsense/k12-zh.jsonl',
        reader_cfg=k12_reader_cfg,
        infer_cfg=k12_infer_cfg_zh,
        eval_cfg=k12_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='k12-en',
        path='data/language_quality/commonsense/k12-en.jsonl',
        reader_cfg=k12_reader_cfg,
        infer_cfg=k12_infer_cfg_en,
        eval_cfg=k12_eval_cfg)
]
