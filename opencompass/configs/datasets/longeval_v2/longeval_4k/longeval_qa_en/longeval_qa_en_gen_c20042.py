from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, longeval_2wikimqa_eDataset

longeval_qa_en_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
)

longeval_qa_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128)
)

longeval_qa_en_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator),
    pred_role='BOT'
)

longeval_qa_en_datasets = [
    dict(
        abbr='qa_en_4k',
        type=longeval_2wikimqa_eDataset,
        path='./data/longeval_v2/4k/qa_en.json',
        reader_cfg=longeval_qa_en_reader_cfg,
        infer_cfg=longeval_qa_en_infer_cfg,
        eval_cfg=longeval_qa_en_eval_cfg)
]
