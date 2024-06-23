from opencompass.datasets.language_quality.language_quality import ICLEvaluator, ICLDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

icl_en_datasets = []
for k in [0, 4]:
    icl_en_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='dev')

    if k == 0:
        icl_en_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='Answer these questions, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A:'),
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50)
        )
    else:
        icl_en_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='Answer the question, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A: The answer is {answer}.\n'),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>",
                    round=[
                        dict(role='HUMAN', prompt='Answer the question, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A:'),
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=list(range(k))),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

    icl_en_eval_cfg = dict(evaluator=dict(type=ICLEvaluator), pred_role="BOT")

    icl_en_datasets.append(
        dict(
            type=ICLDataset,
            abbr='icl_en' if k == 0 else f'icl_en_{k}shot',
            path='./data/language_quality/logic/icl/',
            reader_cfg=icl_en_reader_cfg,
            infer_cfg=icl_en_infer_cfg,
            eval_cfg=icl_en_eval_cfg)
    )
