from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import TEvalEvaluator
from opencompass.datasets import teval_postprocess, TEvalP10Dataset

plugin_eval_subject_mapping = {
    "instruct_zh": ["instruct_v1_zh"],
    "plan_zh": ["plan_json_v1_zh", "plan_str_v1_zh"],
    "review_zh": ["review_str_v1_zh"],
    "reason_retrieve_understand_zh": ["reason_retrieve_understand_json_v1_zh"],
    "reason_zh": ["reason_str_v1_zh"],
    "retrieve_zh": ["retrieve_str_v1_zh"],
    "understand_zh": ["understand_str_v1_zh"],
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
                abbr='plugin_eval-p10-' + subset,
                type=TEvalP10Dataset,
                path='./data/pluginEval',
                name=subset,
                reader_cfg=plugin_eval_reader_cfg,
                infer_cfg=plugin_eval_infer_cfg,
                eval_cfg=plugin_eval_eval_cfg
            )
        )

del _name, subset
