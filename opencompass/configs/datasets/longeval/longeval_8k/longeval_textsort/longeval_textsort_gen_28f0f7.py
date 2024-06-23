from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import longeval_textsortDataset, longeval_textsortEvaluator
from opencompass.utils.text_postprocessors import general_cn_postprocess

meta_instruction = """
    You are an AI assistant. Your job is to sort multiple book sections into the correct order.
    Each time, you will be provided with 4 pieces of text.
    These texts form a continuous part of a book, but are provided in random order.
    You need to find the correct order and return the answer in a string.
    For example, if you output [4, 1, 3, 2], that means the correct order is: Part 4 -> Part 1 -> Part 3 -> Part 2.
    You will also be provided with the neighboring paragraphs before and after the 4 pieces of texts. \n
    The case sample is shown below and you should give me the answer in the format exactly the same as the sample. \n
    However, you should NOT focus on the content of sample answer. \n
    Please do NOT output any extra content.
    Sample Input (format only): \n
    Before: XXX (Text before the continuous book part)\n\n
    Part 1: XXX\n\n
    Part 2: XXX\n\n
    Part 3: XXX\n\n
    Part 4: XXX\n\n
    After: XXX (Text after the continuous book part)\n\n
    Sample Output (format only): \n
    Answer: [4, 1, 3, 2] \n\n
"""

longeval_textsort_reader_cfg = dict(
    input_columns=['before', 'part1', 'part2', 'part3', 'part4', 'after'],
    output_column='answer')

longeval_textsort_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt= meta_instruction + """
    The question you need to answer is given below.
    Input:\n
    Problem:\n
    Before: {before}\n\n
    Part 1: {part1}\n\n
    Part 2: {part2}\n\n
    Part 3: {part3}\n\n
    Part 4: {part4}\n\n
    After: {after}\n\n
    Now you need to find the correct order of these parts.
    Output:\n
    """),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, max_out_len=128, max_seq_len=16384))

longeval_textsort_eval_cfg = dict(
    evaluator=dict(type=longeval_textsortEvaluator),
    pred_role='BOT'
)

longeval_textsort_datasets = [
    dict(
        abbr='textsort_8k',
        type=longeval_textsortDataset,
        path='./data/longeval/8k/textsort.json',
        reader_cfg=longeval_textsort_reader_cfg,
        infer_cfg=longeval_textsort_infer_cfg,
        eval_cfg=longeval_textsort_eval_cfg)
]
