from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CriticBenchDataset, CriticBenchEvaluator
from mmengine.config import read_base


with read_base():
     from .formats import mappings
     from .prompts import prompts


cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev',
    test_split='dev')


eval_cfg = dict(
    evaluator=dict(type=CriticBenchEvaluator), key='ENV')


set_names = [
    'translate',
    'qa',
    'chat',
    'summary',
    'harmlessness',
    'math_cot',
    'math_pot',
    'code_exec',
    'code_not_exec'
]


criticbench_datasets = []
for mode_name in ['feedback', 'comp_feedback', 'correction', 'meta_feedback']:
    for set_name in set_names:
        # True for subjective and False for objective
        flags = ['obj'] if mode_name in ['correction', 'meta_feedback'] else ['sub', 'obj']
        for flag in flags:
            if mode_name in ['feedback', 'comp_feedback']:
                prefix_prompt = prompts[mode_name][flag][set_name]['prefix_prompt']
                post_prompt = prompts[mode_name][flag][set_name]['post_prompt']
            else:
                prefix_prompt = prompts[mode_name][set_name]['prefix_prompt']
                post_prompt = prompts[mode_name][set_name]['post_prompt']

            infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(
                                role="HUMAN", 
                                prompt=prefix_prompt + "{question}" + post_prompt
                            )
                        ]
                    )
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            )

            if mode_name in ['feedback', 'correction']:
                data_path = f'{set_name}_feedback_correction.json'
            elif mode_name  == 'comp_feedback':
                data_path = f'{set_name}_comp_feedback.json'
            else:
                data_path = f'{set_name}_meta_feedback.json'

            criticbench_datasets.append(dict(
                abbr=f'{set_name}_{mode_name}_{flag}',
                type=CriticBenchDataset,
                path='./data/criticbench_v0.8',
                name=data_path,
                flag_name=flag,
                domain_name=set_name,
                mode_name=mode_name,
                mappings=mappings,
                reader_cfg=cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_cfg),
            )
        # append the new objective dataset with reversed position
        if mode_name == 'comp_feedback':
            criticbench_datasets.append(dict(
                abbr=f'{set_name}_{mode_name}_obj_reverse',
                type=CriticBenchDataset,
                path='./data/criticbench_v0.8',
                reverse=True,
                name=data_path,
                flag_name=flag,
                domain_name=set_name,
                mode_name=mode_name,
                mappings=mappings,
                reader_cfg=cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_cfg),
            )

