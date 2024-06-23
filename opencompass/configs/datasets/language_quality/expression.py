from mmengine.config import read_base

with read_base():
    from .expression.emotion_gen import emotion_datasets
    from .expression.informative import informative_base_datasets, informative_chat_datasets
    from .expression.rule_following import rule_following_base_datasets, rule_following_chat_datasets
    from .expression.text2word_gen import text2word_datasets
    from .expression.word_diversity import word_diversity_base_datasets, word_diversity_chat_datasets

shared = emotion_datasets + text2word_datasets
expression_base = shared + informative_base_datasets + rule_following_base_datasets + word_diversity_base_datasets
expression_api = shared + informative_chat_datasets + rule_following_chat_datasets + word_diversity_chat_datasets
