import json
import random

from datasets import Dataset, load_dataset

from .base import BaseDataset


def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


def get_sample(val: int):
    tmp = [i for i in range(val - 3, val + 4) if i != val]
    return random.sample(tmp, 3)


class gsm8k_noise_dataset_reader(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        # random add noise value to question
        def process(example):
            answer_val = int(gsm8k_dataset_postprocess(example['answer']))
            extra_options = get_sample(answer_val)
            pos = random.sample(list(range(4)), 1)[0]
            extra_options.insert(pos, answer_val)
            example['options'] = [str(i) for i in extra_options]
            example['answer'] = chr(ord('A') + pos)
            return example

        dataset = dataset.map(process)
        return dataset


class GSM8KReferenceSkywork(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                dataset.append(line)
        return Dataset.from_list(dataset)
