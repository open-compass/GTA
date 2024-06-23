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
    inferencer=dict(type=GenInferencer, max_out_len=2048))

subjectiveqa_eval_cfg = dict(
    evaluator=dict(type=SubjectiveQAEvaluator), pred_role='BOT')

labinfo_datasets = [
    dict(
        type=SubjectiveQAv3,
        abbr='labinfo',
        path='/cpfs01/shared/public/public_hdd/lishuaibin/data_sub/',
        file_name='lab_22.jsonl',
        reader_cfg=subjectiveqa_reader_cfg,
        infer_cfg=subjectiveqa_infer_cfg,
        eval_cfg=subjectiveqa_eval_cfg)
]