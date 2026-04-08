import os

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer

from opencompass.datasets.gta_bench_v2 import GTABenchDataset, GPTEvaluator


# end模式配置
end_reader_cfg = dict(
    input_columns=["dialogs", "resources", "full_tree"],
    output_column=None,
    train_split='test',
    test_split='test')


gta_bench_infer_cfg_end = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every'),
)

gta_bench_eval_cfg_end = dict(
    evaluator=dict(type=GPTEvaluator, mode='every', proxy=os.getenv('EVAL_PROXY')))

gta_bench_datasets = [
    dict(
        abbr="gta_bench_end",
        type=GTABenchDataset,
        path="data/gta_dataset_v2",
        reader_cfg=end_reader_cfg,
        infer_cfg=gta_bench_infer_cfg_end,
        eval_cfg=gta_bench_eval_cfg_end,
        mode='end',
    ),
]
