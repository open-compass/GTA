from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer

from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator
from opencompass.datasets import CIBenchEvaluator

gta_bench_reader_cfg = dict(
    input_columns=["dialogs", "resources"],
    output_column="gt_answer",
    train_split='test',
    test_split='test')

gta_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt'),
)


gta_bench_eval_cfg = dict(evaluator=dict(type=GTABenchEvaluator, mode='every_with_gt'))

gta_bench_datasets = [
    dict(
        abbr="gta_bench",
        type=GTABenchDataset,
        path="data/gta_dataset",
        reader_cfg=gta_bench_reader_cfg,
        infer_cfg=gta_bench_infer_cfg,
        eval_cfg=gta_bench_eval_cfg,
    )
]
