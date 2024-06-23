import unittest
from unittest.mock import Mock

from opencompass.openicl import DatasetReader
from opencompass.openicl.icl_retriever.icl_base_retriever import BaseRetriever


class TestBaseRetriever(unittest.TestCase):

    def setUp(self):
        self.dataset_reader = Mock(spec=DatasetReader)
        self.ice_separator = '\n'
        self.ice_eos_token = '\n'
        self.prompt_eos_token = ''
        self.ice_num = 1
        self.index_split = 'train'
        self.test_split = 'test'
        self.accelerator = None
        self.base_retriever = BaseRetriever(self.dataset_reader,
                                            self.ice_separator,
                                            self.ice_eos_token,
                                            self.prompt_eos_token,
                                            self.ice_num, self.index_split,
                                            self.test_split, self.accelerator)

    def test_init(self):
        # Test case where all arguments are valid
        self.assertIsInstance(self.base_retriever, BaseRetriever)

        # Test case where dataset_reader is not a DatasetReader object
        with self.assertRaises(AttributeError):
            BaseRetriever('invalid_dataset_reader')

        # Test case where index_ds and test_ds are not set correctly
        dataset_reader = Mock(spec=DatasetReader)
        dataset_reader.dataset = {'train': 'train_ds', 'test': 'test_ds'}
        with self.assertRaises(AttributeError):
            BaseRetriever(dataset_reader)

    def test_retrieve(self):
        # Test case where test_ds is a Dataset object
        dataset_reader = Mock(spec=DatasetReader)
        dataset_reader.dataset = 'test_ds'
        base_retriever = BaseRetriever(dataset_reader)
        self.assertEqual(base_retriever.retrieve(), [])

        # Test case where test_ds is a DatasetDict object
        dataset_reader = Mock(spec=DatasetReader)
        dataset_reader.dataset = {'train': 'train_ds', 'test': 'test_ds'}
        base_retriever = BaseRetriever(dataset_reader)
        self.assertEqual(base_retriever.retrieve(), [])


if __name__ == '__main__':
    unittest.main()
