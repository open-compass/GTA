from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, longeval_gov_report_eDataset

longeval_summarization_en_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
)

longeval_summarization_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are given a document. Write a one-page summary of the document.\n\nDocument:\n{context}\n\nNow, write a one-page summary of the document.\n\nSummary:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

longeval_summarization_en_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator),
    pred_role='BOT'
)

longeval_summarization_en_datasets = [
    dict(
        abbr='summarization_en_15k',
        type=longeval_gov_report_eDataset,
        path='./data/longeval_v2/15k/summarization_en.json',
        reader_cfg=longeval_summarization_en_reader_cfg,
        infer_cfg=longeval_summarization_en_infer_cfg,
        eval_cfg=longeval_summarization_en_eval_cfg)
]
