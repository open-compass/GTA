import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--file_name',
                        type=str,
                        default='val-examples500-subjective')
    parser.add_argument('--nums', type=int, default=1)
    parser.add_argument('--type', type=str, default='v7')
    args = parser.parse_args()

    assert args.type in ['v7', 'v13']
    root_path = args.root_path
    if args.nums == 1:
        files = [f'{args.file_name}.json']
    else:
        files = [f'{args.file_name}_{i}.json' for i in range(args.nums)]

    # files = ["val-examples200-subjective.json"]

    saving_path = osp.join(root_path.replace('/predictions', '/summary'),
                           f'{args.file_name}.xlsx')

    outputs = defaultdict(list)
    model_list = list(os.listdir(root_path))

    questions = []
    question_finish = False
    for model_name in model_list:
        output_list = []
        for file_name in files:
            file_path = osp.join(root_path, model_name, file_name)
            single_out_dict = mmengine.load(file_path)
            for item, value in single_out_dict.items():
                suffix = ''
                if not question_finish:
                    question = value['origin_prompt']
                    if '<|Human|>െ' in question:
                        human_prefix = '<|Human|>െ'
                        assistant_prefix = '<|Assistant|>െ'
                    elif '<|User|>:' in question:
                        human_prefix = '<|User|>:'
                        assistant_prefix = '<|Assistant|>:'
                    elif '<TOKENS_UNUSED_140>user\n' in question:
                        human_prefix = '<TOKENS_UNUSED_140>user\n'
                        assistant_prefix = '<TOKENS_UNUSED_140>assistant\n'
                        suffix = '<TOKENS_UNUSED_139>'
                    elif '[UNUSED_TOKEN_146]user\n' in question:
                        human_prefix = '[UNUSED_TOKEN_146]user\n'
                        assistant_prefix = '[UNUSED_TOKEN_146]assistant\n'
                        suffix = '[UNUSED_TOKEN_145]'
                    else:
                        raise ValueError('Unknown chat template')
                    question = question.split(human_prefix)[-1].replace(
                        assistant_prefix, '').strip()
                    if suffix != '':
                        question = question.replace(suffix, '').strip()
                    questions.append(question)

                prediction = value['prediction']
                print('prediction', prediction)
                print('--------------')
                # question = question.replace('\n', '<回车>')
                # prediction = prediction.replace('\n', '<回车>')
                output_list.append(prediction)
        question_finish = True
        outputs['question'] = questions
        print()
        outputs[model_name] = output_list

    df = pd.DataFrame(outputs)
    df = df.to_excel(saving_path, index=False)
    print('excel文件已保存')
