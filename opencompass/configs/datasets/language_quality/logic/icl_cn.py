from opencompass.datasets.language_quality.language_quality import ICL_CNDataset, ICL_CNEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

icl_cn_datasets = []
for k in [0, 4]:
    icl_cn_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='dev')

    if k == 0:
        icl_cn_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt="请回答问题，你的回答应尽可能简单，用'答案是'作为你的回答的开头。\n问： {question}?"),
                        dict(role='BOT', prompt='答：'),
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50)
        )
    else:
        icl_cn_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt="请回答问题，你的回答应尽可能简单，用'答案是'作为你的回答的开头。\n问： {question}?"),
                        dict(role='BOT', prompt='答: 答案是 {answer}。\n'),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>",
                    round=[
                        dict(role='HUMAN', prompt="请回答问题，你的回答应尽可能简单，用'答案是'作为你的回答的开头。\n问： {question}?"),
                        dict(role='BOT', prompt='答：'),
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=list(range(k))),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

    icl_cn_eval_cfg = dict(evaluator=dict(type=ICL_CNEvaluator), pred_role="BOT")

    icl_cn_datasets.append(
        dict(
            type=ICL_CNDataset,
            abbr='icl_cn' if k == 0 else f'icl_cn_{k}shot',
            path='./data/language_quality/logic/icl/',
            reader_cfg=icl_cn_reader_cfg,
            infer_cfg=icl_cn_infer_cfg,
            eval_cfg=icl_cn_eval_cfg)
    )
