from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import JsonlDataset
from opencompass.datasets.language_quality.language_quality import first_option_postprocess_with_gold

story_reader_cfg = dict(
    input_columns=['story', 'choice1', 'choice2'],
    output_column='answer')

story_infer_cfg_zh = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="请为下面的故事选择一个正确的结局。\n{story}\nA. {choice1}\nB. {choice2}\n答案："
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

story_infer_cfg_en = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="Please choose the correct ending of the following story.\n{story}\nA. {choice1}\nB. {choice2}\nAnwser:"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

story_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(
        type=first_option_postprocess_with_gold,
        options='AB',
        gold_options={'A': 1, 'B': 2}
    ),
)

story_gen_datasets = [
    dict(
        type=JsonlDataset,
        abbr='story-zh',
        path='data/language_quality/commonsense/story-zh.jsonl',
        reader_cfg=story_reader_cfg,
        infer_cfg=story_infer_cfg_zh,
        eval_cfg=story_eval_cfg),
    dict(
        type=JsonlDataset,
        abbr='story-en',
        path='data/language_quality/commonsense/story-en.jsonl',
        reader_cfg=story_reader_cfg,
        infer_cfg=story_infer_cfg_en,
        eval_cfg=story_eval_cfg)
]
