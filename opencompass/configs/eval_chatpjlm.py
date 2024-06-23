from mmengine.config import read_base

with read_base():
    from .summarizers.medium import summarizer
    from .lark import lark_bot_url
    from .datasets.collections.chat_medium import datasets
    from .models.collections.week_3 import models
