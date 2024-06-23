from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, longeval_gov_report_eDataset

longeval_gov_report_e_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
)

longeval_gov_report_e_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

longeval_gov_report_e_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator),
    pred_role='BOT'
)

longeval_gov_report_e_datasets = [
    dict(
        abbr='gov_report_8k',
        type=longeval_gov_report_eDataset,
        path='./data/longeval/8k/gov_report_e.json',
        reader_cfg=longeval_gov_report_e_reader_cfg,
        infer_cfg=longeval_gov_report_e_infer_cfg,
        eval_cfg=longeval_gov_report_e_eval_cfg)
]
