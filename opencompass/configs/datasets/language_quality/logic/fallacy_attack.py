from opencompass.datasets import JsonlDataset

from opencompass.datasets.language_quality.language_quality import fallacy_attack_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator

prompt_cn = """
请评估以下两段句子，其中第二段是第一段的延续。第一段句子开头为：“起始”，第二段句子开头为“后续”。
请重点判断下述句子是否符合常理以及事实，以及第二段的回答有没有延续第一段的内容。
请以JSON格式返回响应，其中包含特定的键和值。
要求：
1. 'score'：句子是否符合事实，值是0到10之间的整数。
2. 'relative'：句子是否正面回答了，如果正面回答，就返回True,否则就返回False。
2. 'reason'：值是一个解释评分理由的字符串。
如果句子为空，请将其评分为0。
"""
prompt_en = """
Please evaluate the following two sentences, the second of which is a continuation of the first. The first sentence begins with "Starting", and the second sentence begins with "Following".
Please focus on determining whether the following sentences are consistent with common sense and facts, and whether the second paragraph continues the content of the first paragraph.
Please return the response in JSON format, which contains specific keys and values.
Requirements:
1. 'score': Whether the sentence is consistent with the facts, the value is an integer between 0 and 10.
2. 'relative': Whether the sentence has answered positively, if it has answered positively, return True, otherwise return False.
3. 'reason': The value is a string explaining the reason for the score.
If the sentence is empty, please rate it as 0.
"""
fallacy_attack_reader_cfg = dict(
    input_columns=["prompt","annotation"],output_column= "label"
)

fallacy_attack_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=30),
)

fallacy_attack_cn_eval_cfg = dict(
    pred_postprocessor=dict(type=fallacy_attack_postprocess),
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        fallacy_role="HUMAN",
                        prompt=prompt_cn
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="起始：{prompt}。后续：{prediction}")]))),
    pred_role="BOT",
)

fallacy_attack_en_eval_cfg = dict(
    pred_postprocessor=dict(type=fallacy_attack_postprocess),
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role="SYSTEM",
                        fallacy_role="HUMAN",
                        prompt=prompt_en
                    ),
                ],
                round=[dict(role="HUMAN",
                            prompt="Starting:{prompt}. Following:{prediction}")]))),
    pred_role="BOT",
)


fallacy_attack_datasets=[
    dict(
        abbr="fallacy_attack_cn",
        type=JsonlDataset,
        path='./data/language_quality/logic/fallacy_attack_cn.jsonl',
        reader_cfg=fallacy_attack_reader_cfg,
        infer_cfg=fallacy_attack_infer_cfg,
        eval_cfg=fallacy_attack_cn_eval_cfg,
    ),
    dict(
        abbr="fallacy_attack_en",
        type=JsonlDataset,
        path='./data/language_quality/logic/fallacy_attack_en.jsonl',
        reader_cfg=fallacy_attack_reader_cfg,
        infer_cfg=fallacy_attack_infer_cfg,
        eval_cfg=fallacy_attack_en_eval_cfg,
    )
]

