from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import longeval_linesDataset, longeval_linesEvaluator

longeval_lines_reader_cfg = dict(
    input_columns=['prompt', 'random_idx'],
    output_column='expected_number')

longeval_lines_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="{prompt} Line <{random_idx}>: <REGISTER_CONTENT> is"),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, max_out_len=300, max_seq_len=16384))

longeval_lines_eval_cfg = dict(
    evaluator=dict(type=longeval_linesEvaluator),
    pred_role='BOT',)

longeval_lines_datasets = [
    dict(
        abbr='lines_2k',
        type=longeval_linesDataset,
        path='./data/longeval/2k/lines.jsonl',
        reader_cfg=longeval_lines_reader_cfg,
        infer_cfg=longeval_lines_infer_cfg,
        eval_cfg=longeval_lines_eval_cfg)
]
