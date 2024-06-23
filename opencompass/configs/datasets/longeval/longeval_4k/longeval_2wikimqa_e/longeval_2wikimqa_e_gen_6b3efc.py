from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, longeval_2wikimqa_eDataset

longeval_2wikimqa_e_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
)

longeval_2wikimqa_e_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32)
)

longeval_2wikimqa_e_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator),
    pred_role='BOT'
)

longeval_2wikimqa_e_datasets = [
    dict(
        abbr='2wikimqa_e_4k',
        type=longeval_2wikimqa_eDataset,
        path='./data/longeval/4k/2wikimqa_e.json',
        reader_cfg=longeval_2wikimqa_e_reader_cfg,
        infer_cfg=longeval_2wikimqa_e_infer_cfg,
        eval_cfg=longeval_2wikimqa_e_eval_cfg)
]
