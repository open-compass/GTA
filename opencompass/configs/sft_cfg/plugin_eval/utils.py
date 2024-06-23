import os
import codecs
import json
import pandas as pd

def json2csv(root_dir, write_to, data_version="v2"):
    output_keys_v1 = [
        "plugin_eval-instruct_v1-string_format_metric",
        "plugin_eval-instruct_v1-string_args_em_metric",
        "plugin_eval-instruct_v1-json_format_metric",
        "plugin_eval-instruct_v1-json_args_em_metric",
        "plugin_eval-plan_json_v1-precision",
        "plugin_eval-plan_json_v1-recall",
        "plugin_eval-plan_json_v1-f1_score",
        "plugin_eval-plan_json_v1-parse_rate",
        "plugin_eval-plan_str_v1-precision",
        "plugin_eval-plan_str_v1-recall",
        "plugin_eval-plan_str_v1-f1_score",
        "plugin_eval-plan_str_v1-parse_rate",
        "plugin_eval-reason_json_v1-thought_bertscore",
        "plugin_eval-reason_json_v1-api_name_metric",
        "plugin_eval-reason_json_v1-api_args_metric",
        "plugin_eval-reason_json_v1-parse_rate",
        "plugin_eval-reason_str_v1-thought_bertscore",
        "plugin_eval-reason_str_v1-api_name_metric",
        "plugin_eval-reason_str_v1-api_args_metric",
        "plugin_eval-reason_str_v1-parse_rate",
        "plugin_eval-retrieval_pool10_v1-precision-json",
        "plugin_eval-retrieval_pool10_v1-recall-json",
        "plugin_eval-retrieval_pool10_v1-f1_score-json",
        "plugin_eval-retrieval_pool10_v1-parse_rate-json",
        "plugin_eval-retrieval_pool10_v1-precision-str",
        "plugin_eval-retrieval_pool10_v1-recall-str",
        "plugin_eval-retrieval_pool10_v1-f1_score-str",
        "plugin_eval-retrieval_pool10_v1-parse_rate-str",
        "plugin_eval-review_json_v1-review_quality",
        "plugin_eval-review_json_v1-parse_rate",
        "plugin_eval-review_str_v1-review_quality",
        "plugin_eval-review_str_v1-parse_rate"
    ]
    output_keys_v2 = [
        "plugin_eval-instruct_v1-string_format_metric",
        "plugin_eval-instruct_v1-string_args_em_metric",
        "plugin_eval-instruct_v1-json_format_metric",
        "plugin_eval-instruct_v1-json_args_em_metric",
        "plugin_eval-plan_json_v1-precision",
        "plugin_eval-plan_json_v1-recall",
        "plugin_eval-plan_json_v1-f1_score",
        "plugin_eval-plan_json_v1-parse_rate",
        "plugin_eval-plan_str_v1-precision",
        "plugin_eval-plan_str_v1-recall",
        "plugin_eval-plan_str_v1-f1_score",
        "plugin_eval-plan_str_v1-parse_rate",
        "plugin_eval-reason_retrieve_understand_json_v2-thought",
        "plugin_eval-reason_retrieve_understand_json_v2-name",
        "plugin_eval-reason_retrieve_understand_json_v2-args_precision",
        "plugin_eval-reason_retrieve_understand_json_v2-args_recall",
        "plugin_eval-reason_retrieve_understand_json_v2-args_f1_score",
        "plugin_eval-reason_retrieve_understand_json_v2-parse_rate",
        "plugin_eval-reason_str_v2-thought",
        "plugin_eval-reason_str_v2-parse_rate",
        "plugin_eval-retrieve_str_v2-name",
        "plugin_eval-retrieve_str_v2-parse_rate",
        "plugin_eval-understand_str_v2-args_precision",
        "plugin_eval-understand_str_v2-args_recall",
        "plugin_eval-understand_str_v2-args_f1_score",
        "plugin_eval-understand_str_v2-parse_rate",
        "plugin_eval-review_json_v1-review_quality",
        "plugin_eval-review_json_v1-parse_rate",
        "plugin_eval-review_str_v1-review_quality",
        "plugin_eval-review_str_v1-parse_rate"
    ]
    output_keys = output_keys_v1 if data_version == "v1" else output_keys_v2
    result = {}
    for f in os.listdir(root_dir):
        f_n = f.split(".")[0]
        with codecs.open(os.path.join(root_dir, f), "r", "utf-8") as fr:
            d = json.load(fr)
            for k, v in d.items():
                result[f"{f_n}-{k}"] = v
    result = [[k, result.get(k, None)] for k in output_keys]
    df = pd.DataFrame(result, columns=["metric", "score"])
    df.to_csv(write_to)
    return result


if __name__ == "__main__":
    # root_dir = "outputs/plugin_eval/plugineval_v2/20231127_184909/results"
    root_dir = "outputs/20231127_plugineval_test/20231127_112217/results/"
    all_results = []
    columns = []
    for f in sorted(os.listdir(root_dir)):
        if f.endswith("csv"):
            continue
        temp = json2csv(
            root_dir=os.path.join(root_dir, f), 
            write_to=os.path.join(root_dir, f"{f}.csv")
        )
        all_results.append(temp)
        columns.append(f)
    df = []
    for item in zip(*all_results):
        temp = []
        for i, t in enumerate(item):
            if i == 0:
                temp.extend(t)
            else:
                temp.append(t[1])
        df.append(temp)
    df = pd.DataFrame(df, columns=["metric"] + columns)
    df.to_csv(os.path.join(root_dir, "all_results.csv"))
