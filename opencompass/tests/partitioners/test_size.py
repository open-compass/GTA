import unittest

from mmengine.config import ConfigDict

from opencompass.partitioners.size import SizePartitioner


class TestSizePartitioner(unittest.TestCase):

    def setUp(self):
        self.max_task_size = 500
        self.gen_task_coef = 1
        self.partition = SizePartitioner('tmp/', self.max_task_size,
                                         self.gen_task_coef)
        self.get_cost

    def test_get_cost(self):
        dataset_cfg = ConfigDict({
            'name': 'test_dataset',
            'eval_cfg': {
                'ds_split': 'test'
            },
            'infer_cfg': {
                'prompt_template': {
                    'template': 'This is a test prompt template.'
                },
                'retriever': {
                    'test_split': 'test'
                }
            }
        })
        cost = self.partition.get_cost(dataset_cfg)
        self.assertEqual(cost, 10)

    def test_split_dataset(self):
        dataset_cfg = ConfigDict({
            'name': 'test_dataset',
            'eval_cfg': {
                'ds_split': 'test'
            },
            'infer_cfg': {
                'prompt_template': {
                    'template': 'This is a test prompt template.'
                },
                'retriever': {
                    'test_split': 'test'
                }
            }
        })
        split_configs = self.partition.split_dataset(dataset_cfg)
        self.assertEqual(len(split_configs), 1)
        self.assertEqual(split_configs[0]['name'], 'test_dataset_0')
        self.assertEqual(split_configs[0]['infer_cfg']['reader']['ds_size'],
                         'test[0:10]')

    def test_partition(self):
        models = [
            ConfigDict(dict(abbr='model1')),
            ConfigDict(dict(abbr='model2'))
        ]
        datasets = [
            ConfigDict({
                'name': 'test_dataset1',
                'eval_cfg': {
                    'ds_split': 'test'
                },
                'infer_cfg': {
                    'prompt_template': {
                        'template': 'This is a test prompt template.'
                    },
                    'retriever': {
                        'test_split': 'test'
                    }
                }
            }),
            ConfigDict({
                'name': 'test_dataset2',
                'eval_cfg': {
                    'ds_split': 'test'
                },
                'infer_cfg': {
                    'prompt_template': {
                        'template': 'This is another test prompt template.'
                    },
                    'retriever': {
                        'test_split': 'test'
                    }
                }
            })
        ]
        tasks = self.partition.partition(models, datasets, 'work_dir',
                                         'out_dir')
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(tasks[0]['datasets'][0]), 1)
        self.assertEqual(tasks[0]['datasets'][0][0]['name'], 'test_dataset1')
        self.assertEqual(len(tasks[1]['datasets'][0]), 1)
        self.assertEqual(tasks[1]['datasets'][0][0]['name'], 'test_dataset2')
