from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SubjectiveQAv3, SubjectiveQAEvaluator

subjectiveqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

subjectiveqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

subjectiveqa_eval_cfg = dict(
    evaluator=dict(type=SubjectiveQAEvaluator), pred_role='BOT')

# 主要改动地方在这里：
# 如果需要 改动 prompt 或者 fewshot 等，根据自己需求对上面 infer_cfg 更改。
zh_en_enhance_datasets = [
    dict(
        type=SubjectiveQAv3,
        abbr='zh_cn_enhance',
        path='./data/zh_en_enhance/',
        file_name='zh_en_data.jsonl',
        reader_cfg=subjectiveqa_reader_cfg,
        infer_cfg=subjectiveqa_infer_cfg,
        eval_cfg=subjectiveqa_eval_cfg)
]