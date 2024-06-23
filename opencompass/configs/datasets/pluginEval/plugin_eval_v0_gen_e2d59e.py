from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import TEvalEvaluator
from opencompass.datasets import TEvalDataset

plugin_eval_subject_mapping = {
    "instruct": ["instruct"],
    "reason": ["reasoning"],
    "plan": ["planning_ReWOO_v4", "planning_json_v2"],
    "retrieval": [
        "api_retrieval_0", "api_retrieval_2", "api_retrieval_4",
        "api_retrieval_6", "api_retrieval_x"
    ]
}

plugin_eval_reader_cfg = dict(
    input_columns=['system_prompt', 'user_input'],
    output_column='ground_truth')

plugin_eval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="{system_prompt}\n{user_input}"),
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
        )
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
