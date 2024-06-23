from mmengine.config import read_base

with read_base():
    from .summarizers.medium import summarizer
    from .lark import lark_bot_url
    from .datasets.collections.base_small import datasets
    from .models.classic.pjlm-0.2 import models
