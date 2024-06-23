import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='process subjectiveqa v3 outputs to excel')
    parser.add_argument('root_path', help='prediction results path')
    parser.add_argument('--saving_name',
                        default='subjectiveqav3.xlsx',
                        help='source data dir')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root_path = args.root_path
    files = ['subjectiveqav3_0.json', 'subjectiveqav3_1.json']

    saving_name = args.saving_name
    saving_path = osp.join(root_path, saving_name)

    outputs = defaultdict(list)
    model_list = list(os.listdir(root_path))

    for model_name in model_list:
        # if is not a dir, skip
        if not osp.isdir(osp.join(root_path, model_name)):
            continue
        output_list = []
        for file_name in files:
            file_path = osp.join(root_path, model_name, file_name)
            single_out_dict = mmengine.load(file_path)
            for item, value in single_out_dict.items():
                prediction = value['prediction']
                print('prediction', prediction)
                print('--------------')
                # question = question.replace('\n', '<回车>')
                # prediction = prediction.replace('\n', '<回车>')
                output_list.append(prediction)
        print()
        outputs[model_name] = output_list

    df = pd.DataFrame(outputs)
    df = df.to_excel(saving_path, index=False)
    print('excel文件已保存')


if __name__ == '__main__':
    main()
