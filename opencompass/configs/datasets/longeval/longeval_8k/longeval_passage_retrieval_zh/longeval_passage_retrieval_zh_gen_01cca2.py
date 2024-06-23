from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRetrievalEvaluator, longeval_passage_retrieval_zhDataset

longeval_passage_retrieval_zh_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
)

longeval_passage_retrieval_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32)
)

longeval_passage_retrieval_zh_eval_cfg = dict(
    evaluator=dict(type=LongBenchRetrievalEvaluator, language='zh'),
    pred_role='BOT'
)

longeval_passage_retrieval_zh_datasets = [
    dict(
        abbr='passage_retrieval_zh_8k',
        type=longeval_passage_retrieval_zhDataset,
        path='./data/longeval/8k/passage_retrieval_zh.json',
        reader_cfg=longeval_passage_retrieval_zh_reader_cfg,
        infer_cfg=longeval_passage_retrieval_zh_infer_cfg,
        eval_cfg=longeval_passage_retrieval_zh_eval_cfg)
]
