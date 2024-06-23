from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import TEvalEvaluator
from opencompass.datasets import teval_postprocess, TEvalDataset

plugin_eval_subject_mapping = {
    "instruct": ["instruct_v1"],
    "reason": ["reason_json_v1", "reason_str_v1"],
    "plan": ["plan_json_v1", "plan_str_v1"],
    "retrieval": ["retrieval_pool10_v1"],
    "review": ["review_json_v1", "review_str_v1"]
}

plugin_eval_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth')

plugin_eval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="{prompt}"),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer))

plugin_eval_all_sets = list(plugin_eval_subject_mapping.keys())

plugin_eval_datasets = []
for _name in plugin_eval_all_sets:
    plugin_eval_eval_cfg = dict(
        evaluator=dict(
            type=TEvalEvaluator,
            subset=_name
        ),
        pred_postprocessor=dict(type=teval_postprocess)
    )
    for subset in plugin_eval_subject_mapping[_name]:
        plugin_eval_datasets.append(
            dict(
                abbr='plugin_eval-' + subset,
                type=TEvalDataset,
                path='./data/pluginEval',
                name=subset,
                reader_cfg=plugin_eval_reader_cfg,
                infer_cfg=plugin_eval_infer_cfg,
                eval_cfg=plugin_eval_eval_cfg
            )
        )

del _name, subset
