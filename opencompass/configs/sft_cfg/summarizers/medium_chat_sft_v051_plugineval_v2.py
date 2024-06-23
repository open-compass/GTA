from mmengine.config import read_base

with read_base():
    from .medium_chat_sft_v051 import summarizer
    # from ...summarizers.groups.teval import plugin_eval_summary_groups

# summary_groups_plugineval = sum(
#     [v for k, v in locals().items() if k.endswith("_summary_groups")], [])

summarizer["dataset_abbrs"].extend(
    [
        ['plugin_eval-instruct_v1', 'string_format_metric'],
        ['plugin_eval-instruct_v1', 'string_args_em_metric'],
        ['plugin_eval-instruct_v1', 'json_format_metric'],
        ['plugin_eval-instruct_v1', 'json_args_em_metric'],
        ['plugin_eval-plan_json_v1', 'precision'],
        ['plugin_eval-plan_json_v1', 'recall'],
        ['plugin_eval-plan_json_v1', 'f1_score'],
        ['plugin_eval-plan_json_v1', 'parse_rate'],
        ['plugin_eval-plan_str_v1', 'precision'],
        ['plugin_eval-plan_str_v1', 'recall'],
        ['plugin_eval-plan_str_v1', 'f1_score'],
        ['plugin_eval-plan_str_v1', 'parse_rate'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'thought'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'name'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'args_precision'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'args_recall'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'args_f1_score'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'parse_rate'],
        ['plugin_eval-reason_str_v1', 'thought'],
        ['plugin_eval-reason_str_v1', 'parse_rate'],
        ['plugin_eval-retrieve_str_v1', 'name'],
        ['plugin_eval-retrieve_str_v1', 'parse_rate'],
        ['plugin_eval-understand_str_v1', 'args_precision'],
        ['plugin_eval-understand_str_v1', 'args_recall'],
        ['plugin_eval-understand_str_v1', 'args_f1_score'],
        ['plugin_eval-understand_str_v1', 'parse_rate'],
        ['plugin_eval-review_json_v1', 'review_quality'],
        ['plugin_eval-review_json_v1', 'parse_rate'],
        ['plugin_eval-review_str_v1', 'review_quality'],
        ['plugin_eval-review_str_v1', 'parse_rate'],
        ['plugin_eval-instruct_v1_zh', 'string_format_metric'],
        ['plugin_eval-instruct_v1_zh', 'string_args_em_metric'],
        ['plugin_eval-instruct_v1_zh', 'json_format_metric'],
        ['plugin_eval-instruct_v1_zh', 'json_args_em_metric'],
        ['plugin_eval-plan_json_v1_zh', 'precision'],
        ['plugin_eval-plan_json_v1_zh', 'recall'],
        ['plugin_eval-plan_json_v1_zh', 'f1_score'],
        ['plugin_eval-plan_json_v1_zh', 'parse_rate'],
        ['plugin_eval-plan_str_v1_zh', 'precision'],
        ['plugin_eval-plan_str_v1_zh', 'recall'],
        ['plugin_eval-plan_str_v1_zh', 'f1_score'],
        ['plugin_eval-plan_str_v1_zh', 'parse_rate'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'thought'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'name'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'args_precision'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'args_recall'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'args_f1_score'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'parse_rate'],
        ['plugin_eval-reason_str_v1_zh', 'thought'],
        ['plugin_eval-reason_str_v1_zh', 'parse_rate'],
        ['plugin_eval-retrieve_str_v1_zh', 'name'],
        ['plugin_eval-retrieve_str_v1_zh', 'parse_rate'],
        ['plugin_eval-understand_str_v1_zh', 'args_precision'],
        ['plugin_eval-understand_str_v1_zh', 'args_recall'],
        ['plugin_eval-understand_str_v1_zh', 'args_f1_score'],
        ['plugin_eval-understand_str_v1_zh', 'parse_rate'],
        ['plugin_eval-review_json_v1_zh', 'review_quality'],
        ['plugin_eval-review_json_v1_zh', 'parse_rate'],
        ['plugin_eval-review_str_v1_zh', 'review_quality'],
        ['plugin_eval-review_str_v1_zh', 'parse_rate'],
    ]
)

# summarizer["summary_groups"].extend(
#     summary_groups_plugineval
# )
