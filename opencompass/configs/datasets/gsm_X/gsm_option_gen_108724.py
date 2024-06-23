from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess

few_shot_choice = {
    "resoning" : [
                dict(role='HUMAN', prompt='Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA. 6\nB. 7\nC. 8\nD. 10\n'),
                dict(role='BOT', prompt='Answer: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is C. 6.'),
                dict(role='HUMAN', prompt='Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA. 9\nB. 6\nC. 5\nD. 7\n'),
                dict(role='BOT', prompt='Answer: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is C. 5.\n'),
                dict(role='HUMAN', prompt='Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA. 24\nB. 32\nC. 39\nD. 42\n'),
                dict(role='BOT', prompt='Answer: Leah had 32 chocolates and Leah\'s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is C. 39.\n'),
                dict(role='HUMAN', prompt='Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA. 6\nB. 8\nC. 9\nD. 7\n'),
                dict(role='BOT', prompt='Answer: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is B. 8.\n'),
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}\n')],

    "normal" : [
                dict(role='HUMAN', prompt='Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA. 6\nB. 7\nC. 8\nD. 10\n'),
                dict(role='BOT', prompt='The answer is C. 6.\n'),
                dict(role='HUMAN', prompt='Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA. 9\nB. 6\nC. 5\nD. 7\n'),
                dict(role='BOT', prompt='The answer is C. 5.\n'),
                dict(role='HUMAN', prompt='Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA. 24\nB. 32\nC. 39\nD. 42\n'),
                dict(role='BOT', prompt='The answer is C. 39.\n'),
                dict(role='HUMAN', prompt='Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA. 6\nB. 8\nC. 9\nD. 7\n'),
                dict(role='BOT', prompt='The answer is B. 8.\n'),
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='The answer is {answer}\n')]
}

mathbench_sets = {
    'gsm_option': ['single_choice_en']
}


# Generate reasoning path if set True or just generate the final answer
with_reasoning = True

# Use circular evaluation or not
with_circular_eval = True

gsm8k_option_datasets = []

for _split in list(mathbench_sets.keys()):
    for _name in mathbench_sets[_split]:
        mathbench_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=few_shot_choice['resoning'] if with_reasoning else few_shot_choice['normal'],
                    ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512),
        )

        mathbench_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if 'choice' in _name else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD') if 'single_choice' in _name else dict(type=mathbench_postprocess, name=_name))

        gsm8k_option_datasets.append(
            dict(
                abbr="gsm-X-options",
                type=MathBenchDataset,
                path=f"./data/gsm-X/{_split}",
                name=_name,
                with_circular=with_circular_eval,
                reader_cfg=dict(
                    input_columns=["question"],
                    output_column="answer"
                    ),
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            ))
