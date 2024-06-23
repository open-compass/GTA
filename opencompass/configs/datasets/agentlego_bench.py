from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer

from opencompass.datasets.agentlego_bench import AgentLegoBenchDataset, AgentLegoBenchEvaluator
from opencompass.datasets import CIBenchEvaluator

agentlego_bench_reader_cfg = dict(
    input_columns=["dialogs", "resources"],
    output_column="gt_answer",
    train_split='test',
    test_split='test')

agentlego_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt'),
)


agentlego_bench_eval_cfg = dict(evaluator=dict(type=AgentLegoBenchEvaluator, mode='every_with_gt'))

agentlego_bench_datasets = [
    dict(
        abbr="agentlego_bench",
        type=AgentLegoBenchDataset,
        path="data/agentlego_bench_229",
        reader_cfg=agentlego_bench_reader_cfg,
        infer_cfg=agentlego_bench_infer_cfg,
        eval_cfg=agentlego_bench_eval_cfg,
    )
]
