from fractions import Fraction
from unittest import TestCase

from tensorflow.data import Dataset

from boiling_learning.datasets.datasets import (DatasetSplits, bulk_split,
                                                calculate_dataset_size)


class datasets_test(TestCase):
    def test_calculate_size(self):
        ds = Dataset.range(10)
        self.assertEqual(
            calculate_dataset_size(ds),
            10
        )

    def test_bulk_split(self):
        ds = Dataset.range(10)

        ds_train, ds_val, ds_test = bulk_split(
            ds,
            DatasetSplits(train=Fraction(6, 10), val=Fraction(3, 10))
        )

        self.assertSequenceEqual(tuple(ds_train), tuple(Dataset.range(6)))
        self.assertSequenceEqual(tuple(ds_val), tuple(Dataset.range(6, 9)))
        self.assertSequenceEqual(tuple(ds_test), tuple(Dataset.range(9, 10)))
