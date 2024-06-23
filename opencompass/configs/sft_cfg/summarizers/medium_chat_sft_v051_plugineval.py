from mmengine.config import read_base

with read_base():
    from .medium_chat_sft_v051 import summarizer
    # from ...summarizers.groups.teval import plugin_eval_summary_groups

# summary_groups_plugineval = sum(
#     [v for k, v in locals().items() if k.endswith("_summary_groups")], [])

summarizer["dataset_abbrs"].extend(
    [
        ['plugin_eval-api_retrieval_0', 'precision'],
        ['plugin_eval-api_retrieval_0', 'recall'],
        ['plugin_eval-api_retrieval_0', 'f1_score'],
        ['plugin_eval-api_retrieval_0', 'parse_rate'],
        ['plugin_eval-api_retrieval_2', 'precision'],
        ['plugin_eval-api_retrieval_2', 'recall'],
        ['plugin_eval-api_retrieval_2', 'f1_score'],
        ['plugin_eval-api_retrieval_2', 'parse_rate'],
        ['plugin_eval-api_retrieval_4', 'precision'],
        ['plugin_eval-api_retrieval_4', 'recall'],
        ['plugin_eval-api_retrieval_4', 'f1_score'],
        ['plugin_eval-api_retrieval_4', 'parse_rate'],
        ['plugin_eval-api_retrieval_6', 'precision'],
        ['plugin_eval-api_retrieval_6', 'recall'],
        ['plugin_eval-api_retrieval_6', 'f1_score'],
        ['plugin_eval-api_retrieval_6', 'parse_rate'],
        ['plugin_eval-api_retrieval_x', 'precision'],
        ['plugin_eval-api_retrieval_x', 'recall'],
        ['plugin_eval-api_retrieval_x', 'f1_score'],
        ['plugin_eval-api_retrieval_x', 'parse_rate'],
        ['plugin_eval-retrieval', 'precision'],
        ['plugin_eval-retrieval', 'recall'],
        ['plugin_eval-retrieval', 'f1_score'],
        ['plugin_eval-retrieval', 'parse_rate'],
        ['plugin_eval-instruct', 'string_format_metric'],
        ['plugin_eval-instruct', 'string_args_em_metric'],
        ['plugin_eval-instruct', 'json_format_metric'],
        ['plugin_eval-instruct', 'json_args_em_metric'],
        ['plugin_eval-planning_json_v2', 'precision'],
        ['plugin_eval-planning_json_v2', 'recall'],
        ['plugin_eval-planning_json_v2', 'f1_score'],
        ['plugin_eval-planning_ReWOO_v4', 'precision'],
        ['plugin_eval-planning_ReWOO_v4', 'recall'],
        ['plugin_eval-planning_ReWOO_v4', 'f1_score'],
        ['plugin_eval-reasoning', 'thought_metric-bertscore_precision'],
        ['plugin_eval-reasoning', 'thought_metric-bertscore_recall'],
        ['plugin_eval-reasoning', 'thought_metric-bertscore_f1'],
        ['plugin_eval-reasoning', 'api_name_metric-exact_match'],
        ['plugin_eval-reasoning', 'api_args_metric-keys_match-precision'],
        ['plugin_eval-reasoning', 'api_args_metric-keys_match-recall'],
        ['plugin_eval-reasoning', 'api_args_metric-keys_match-f1'],
        ['plugin_eval-reasoning', 'api_args_metric-words_similarity-bertscore_precision'],
        ['plugin_eval-reasoning', 'api_args_metric-words_similarity-bertscore_recall'],
        ['plugin_eval-reasoning', 'api_args_metric-words_similarity-bertscore_f1'],
    ]
)

# summarizer["summary_groups"].extend(
#     summary_groups_plugineval
# )
