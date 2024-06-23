from mmengine.config import read_base

with read_base():
    from .medium_chat_sft_v051 import summarizer
    # from ...summarizers.groups.teval import plugin_eval_summary_groups

# summary_groups_plugineval = sum(
#     [v for k, v in locals().items() if k.endswith("_summary_groups")], [])

summarizer["dataset_abbrs"].extend(
    [
        ['plugin_eval-retrieval_pool10_v1', 'precision-str'],
        ['plugin_eval-retrieval_pool10_v1', 'recall-str'],
        ['plugin_eval-retrieval_pool10_v1', 'f1_score-str'],
        ['plugin_eval-retrieval_pool10_v1', 'parse_rate-str'],
        ['plugin_eval-retrieval_pool10_v1', 'precision-json'],
        ['plugin_eval-retrieval_pool10_v1', 'recall-json'],
        ['plugin_eval-retrieval_pool10_v1', 'f1_score-json'],
        ['plugin_eval-retrieval_pool10_v1', 'parse_rate-json'],
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
        ['plugin_eval-reason_json_v1', 'thought_bertscore'],
        ['plugin_eval-reason_json_v1', 'api_name_metric'],
        ['plugin_eval-reason_json_v1', 'api_args_metric'],
        ['plugin_eval-reason_json_v1', 'parse_rate'],
        ['plugin_eval-reason_str_v1', 'thought_bertscore'],
        ['plugin_eval-reason_str_v1', 'api_name_metric'],
        ['plugin_eval-reason_str_v1', 'api_args_metric'],
        ['plugin_eval-reason_str_v1', 'parse_rate'],
        ['plugin_eval-review_json_v1', 'review_quality'],
        ['plugin_eval-review_json_v1', 'parse_rate'],
        ['plugin_eval-review_str_v1', 'review_quality'],
        ['plugin_eval-review_str_v1', 'parse_rate'],
    ]
)

# summarizer["summary_groups"].extend(
#     summary_groups_plugineval
# )
