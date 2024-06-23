from mmengine.config import read_base

with read_base():
    from .medium_chat_sft_v051 import datasets
    from ...datasets.pluginEval.plugin_eval_v2_gen import (
        plugin_eval_datasets
    )

datasets.extend(plugin_eval_datasets)
