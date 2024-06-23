from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.datasets.language_quality.language_quality import first_number_postprocess_plus

instruction_reader_cfg = dict(
    input_columns=['instruction', 'input', 'output'],
    output_column='category')

instruction_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{instruction} {input}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

instruction_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        prompt="请判断下列回答是否符合指令，可参考标准答案。对回答进行0~100的打分，分数越高表示回答越准确。"
                               "如果回答的语言和指令的语言不一致（翻译指令除外），则直接打分为0分。"
                               "如果回答没有正面回答指令内容，直接打分为0分。"
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="指令：{instruction} {input}\n标准答案：{output}\n回答：{prediction}")])),
        postprocessor=dict(
            type=first_number_postprocess_plus
        )
    ),
    pred_role="BOT",
)

instruction_datasets = [
    dict(
        type=JsonlDataset,
        abbr='instruction-zh',
        path='data/language_quality/commonsense/instruction-zh.jsonl',
        reader_cfg=instruction_reader_cfg,
        infer_cfg=instruction_infer_cfg,
        eval_cfg=instruction_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='instruction-en',
        path='data/language_quality/commonsense/instruction-en.jsonl',
        reader_cfg=instruction_reader_cfg,
        infer_cfg=instruction_infer_cfg,
        eval_cfg=instruction_eval_cfg)
]
