from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SubjectiveQAv2, SubjectiveQAEvaluator

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

subjectiveqav2_datasets = [
    dict(
        type=SubjectiveQAv2,
        abbr='subjectiveqav2',
        path='./data/subjectiveqa/',
        file_name='subjectiveqa_v2.jsonl',
        reader_cfg=subjectiveqa_reader_cfg,
        infer_cfg=subjectiveqa_infer_cfg,
        eval_cfg=subjectiveqa_eval_cfg)
]