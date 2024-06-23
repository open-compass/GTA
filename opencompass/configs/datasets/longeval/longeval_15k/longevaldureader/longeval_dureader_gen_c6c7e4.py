from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, longeval_dureaderDataset

longeval_dureader_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
)

longeval_dureader_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128)
)

longeval_dureader_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator, language='zh'),
    pred_role='BOT'
)

longeval_dureader_datasets = [
    dict(
        type=longeval_dureaderDataset,
        abbr='dureader',
        path='./data/longeval/15k/dureader.json',
        reader_cfg=longeval_dureader_reader_cfg,
        infer_cfg=longeval_dureader_infer_cfg,
        eval_cfg=longeval_dureader_eval_cfg)
]
