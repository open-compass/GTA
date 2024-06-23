from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchClassificationEvaluator, longeval_trec_eDataset, trec_postprocess

longeval_classification_en_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='all_labels',
)

longeval_classification_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64)
)

longeval_classification_en_eval_cfg = dict(
    evaluator=dict(type=LongBenchClassificationEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=trec_postprocess),
)

longeval_classification_en_datasets = [
    dict(
        abbr='classification_en_15k',
        type=longeval_trec_eDataset,
        path='./data/longeval_v2/15k/classification_en.json',
        reader_cfg=longeval_classification_en_reader_cfg,
        infer_cfg=longeval_classification_en_infer_cfg,
        eval_cfg=longeval_classification_en_eval_cfg)
]
