from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess

PROMPT_EN = {
    "FEWSHOT_INSTRUCTION_CLOZE": [
        # example-1
        dict(role="HUMAN", prompt="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?"),
        dict(role="BOT", prompt="210"),
        # example-2
        dict(role="HUMAN", prompt="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?"),
        dict(role="BOT", prompt="140"),
        # example-3
        dict(role="HUMAN", prompt="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?"),
        dict(role="BOT", prompt="146"),
        dict(role="HUMAN", prompt="Question: {question}"),
    ],
    "FEWSHOT_INSTRUCTION_CHOICE": [
        # example-1
        dict(role="HUMAN", prompt="Question: Given point P(-1,4) lies on the graph of the inverse proportionality function $y=\\frac{k}{x}$ (k≠0), what is the value of k? A. $-\\frac{1}{4}$ B. $\\frac{1}{4}$ C. $4$ D. $-4$"),
        dict(role="BOT", prompt="D"),
        # example-2
        dict(role="HUMAN", prompt="Question: The graph of the power function $y=(x)$ passes through the point$ (2, \\dfrac {1}{4}) $, what is the value of $f(-3)$? A. $\\frac{1}{9}$ B. $\\frac{1}{8})=196-x$ C. $\\frac{2}{9}$ D. $\\frac{1}{4}$"),
        dict(role="BOT", prompt="A"),
        # example-3
        dict(role="HUMAN", prompt="Question: If $3 x-y=12$, what is the value of $\\frac{8^{x}}{2^{y}} ?$\nA. The value cannot be determined from the information given.\nB. $2^{12}$\nC. 4\nD. $8^{2}$"),
        dict(role="BOT", prompt="B"),
        # question
        dict(role="HUMAN", prompt="Question: {question}"),
    ],
}

PROMPT_CN = {
    "FEWSHOT_INSTRUCTION_CLOZE": [
        # example-1
        dict(role="HUMAN", prompt="问题: Mark的篮球队得到25个2分球，8个3分球和10个罚球。他们的对手得到2分球的两倍，但3分球和罚球的一半。两队得分的总和是多少？"),
        dict(role="BOT", prompt="210"),
        # example-2
        dict(role="HUMAN", prompt="问题: Bella有两倍于飞盘的弹珠。她还比卡片多20个飞盘。如果她买每种物品多2/5，她会有多少总数的物品，如果她现在有60颗弹珠？"),
        dict(role="BOT", prompt="140"),
        # example-3
        dict(role="HUMAN", prompt="问题: 问题一个有4个水果篮子，前三个篮子里有9个苹果、15个橙子和14个香蕉，第四个篮子里每种水果都少2个。总共有多少水果？"),
        dict(role="BOT", prompt="146"),
        dict(role="HUMAN", prompt="问题: {question}"),
    ],
    "FEWSHOT_INSTRUCTION_CHOICE": [
        # example-1
        dict(role="HUMAN", prompt="问题: 已知点P（-1，4）在反比例函数$y=\\frac{k}{x}$ (k≠0)的图象上，则k的值是____\nA. $-\\frac{1}{4}$ B. $\\frac{1}{4}$ C. $4$ D. $-4$"),
        dict(role="BOT", prompt="D"),
        # example-2
        dict(role="HUMAN", prompt="问题: 幂函数$ y=(x) $的图象经过点$ (2, \\dfrac {1}{4}) $，则$ f(-3) $的值为 ______ ．\nA. $\\frac{1}{9}$ B. $\\frac{1}{8})=196-x$ C. $\\frac{2}{9}$ D. $\\frac{1}{4}$"),
        dict(role="BOT", prompt="A"),
        # example-3
        dict(role="HUMAN", prompt="问题: 如果$3 x-y=12$，则$\\frac{8^{x}}{2^{y}}$的值是多少？\nA. 无法从给定的信息中确定值。\nB. $2^{12}$\nC. 4\nD. $8^{2}$"),
        dict(role="BOT", prompt="B"),
        dict(role="HUMAN", prompt="问题: {question}"),
    ],
}

compassbench_v1_math_native_sets = {
    'high': ['single_choice_cn', 'single_choice_en'],
    'middle': ['single_choice_cn', 'single_choice_en'],
    'primary': ['cloze_cn', 'cloze_en'],
}

# Use circular evaluation or not
with_circular_eval = True

compassbench_v1_math_native_datasets = []

for _split in list(compassbench_v1_math_native_sets.keys()):
    for _name in compassbench_v1_math_native_sets[_split]:
        prompt_example = PROMPT_CN if '_cn' in _name else PROMPT_EN
        mathbench_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=prompt_example["FEWSHOT_INSTRUCTION_CLOZE"] if 'cloze' in _name else prompt_example["FEWSHOT_INSTRUCTION_CHOICE"]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512),
        )

        mathbench_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if 'choice' in _name and with_circular_eval else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD') if 'single_choice' in _name else dict(type=mathbench_postprocess, name=_name))

        compassbench_v1_math_native_datasets.append(
            dict(
                abbr="compassbench_v1_math-" + _split + '-' + _name + '-native',
                type=MathBenchDataset,
                path=f"./data/compassbench_v1_math_agent/{_split}",
                name=_name,
                with_circular=with_circular_eval,
                reader_cfg=dict(
                    input_columns=["question"],
                    output_column="answer"
                ),
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            ))
