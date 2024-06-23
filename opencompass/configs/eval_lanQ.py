from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import lanQDataset, lanQEvaluator


with read_base():
    from .models.hf_internlm.hf_internlm_7b import models

lanQ_paths = [
    './data/lanQ/en_162.csv',
    './data/lanQ/zh_90.csv',
    './data/lanQ/puyu.csv'
]

lanQ_datasets = []
for path in lanQ_paths:
    lanQ_reader_cfg = dict(input_columns="input", output_column=None)
    lanQ_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role='HUMAN', prompt="{input}")])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    lanQ_eval_cfg = dict(evaluator=dict(type=lanQEvaluator), pred_role="BOT")

    lanQ_datasets.append(
        dict(
            abbr='lanQ_' + path.split('/')[-1].split('.csv')[0],
            type=lanQDataset,
            path=path,
            reader_cfg=lanQ_reader_cfg,
            infer_cfg=lanQ_infer_cfg,
            eval_cfg=lanQ_eval_cfg,
        ))

datasets = lanQ_datasets

work_dir = './outputs/eval_lanQ'
