from fractions import Fraction
from random import sample

import numpy as np
import tensorflow as tf

from boiling_learning.datasets.sliceable import (
    SliceableDataset,
    concatenate,
    sliceable_dataset_to_tensorflow_dataset,
)
from boiling_learning.utils.random import random_state


class TestSliceableDataset:
    def test_basics(self) -> None:
        data = [0, 10, 200, 3, 45]

        sds = SliceableDataset(data)

        assert len(sds) == len(data) == 5
        assert list(sds) == data
        assert sds[2] == 200
        assert sds[0] == 0
        assert sds[-1] == 45

    def test_slicing(self) -> None:
        sds = SliceableDataset([0, 10, 200, 3, 45])

        assert isinstance(sds[1:3], SliceableDataset)

        assert list(sds[:]) == [0, 10, 200, 3, 45]
        assert list(sds[:3]) == [0, 10, 200]
        assert list(sds[2:4]) == [200, 3]
        assert list(sds[3:]) == [3, 45]

    def test_masking(self) -> None:
        sds = SliceableDataset([0, 10, 200, 3, 45])

        assert isinstance(sds[[False, True, False, False, True]], SliceableDataset)
        assert list(sds[[False, False, False, False, False]]) == []
        assert list(sds[[False, True, False, False, True]]) == [10, 45]
        assert list(sds[[True, True, True, True, True]]) == [0, 10, 200, 3, 45]

    def test_selecting(self) -> None:
        sds = SliceableDataset([0, 10, 200, 3, 45])

        assert isinstance(sds[[0, 3, 3, -1, 2, -1]], SliceableDataset)
        assert list(sds[[2, 2, 0, -1, 2, -1]]) == [200, 200, 0, 45, 200, 45]
        assert list(sds[[]]) == []

    def test_zip(self) -> None:
        sds1 = SliceableDataset([10, 5, 2, 8])
        sds2 = SliceableDataset('abcd')
        sds3 = SliceableDataset(range(4))

        sds = SliceableDataset.zip(sds1, sds2, sds3)
        assert sds[0] == (10, 'a', 0)
        assert list(sds) == [
            (10, 'a', 0),
            (5, 'b', 1),
            (2, 'c', 2),
            (8, 'd', 3),
        ]

    def test_apply(self) -> None:
        def stringify(sds: SliceableDataset[int]) -> SliceableDataset[str]:
            return SliceableDataset([str(elem) for elem in sds])

        sds = SliceableDataset([3, 1, 4, 1, 5])
        assert list(sds.apply(stringify)) == list(stringify(sds)) == ['3', '1', '4', '1', '5']

    def test_concatenate(self) -> None:
        sds1 = SliceableDataset([4, 3, 2, 1])
        sds2 = SliceableDataset('abcd')
        sds = sds1.concatenate(sds2)

        assert isinstance(sds, SliceableDataset)
        assert list(sds) == [4, 3, 2, 1, 'a', 'b', 'c', 'd']

    def test_enumerate(self) -> None:
        sds = SliceableDataset('abcd')

        assert isinstance(sds.enumerate(), SliceableDataset)
        assert list(sds.enumerate()) == [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]

    def test_map(self) -> None:
        sds = SliceableDataset('abcd')
        assert list(sds.map(str.upper)) == ['A', 'B', 'C', 'D']

    def test_split(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')
        splits = sds.split(5, 0, None, Fraction(1, 4))

        assert ''.join(splits[0]) == 'abcde'
        assert ''.join(splits[1]) == ''
        assert ''.join(splits[2]) == 'fghijklmnopqrst'
        assert ''.join(splits[3]) == 'uvwxyz'

    def test_shuffle(self) -> None:
        data = 'abcdefghijklmnopqrstuvwxyz'
        with random_state(1997):
            shuffled_data = ''.join(sample(data, k=len(data)))

        sds = SliceableDataset(data)
        with random_state(1997):
            shuffled = sds.shuffle()

        assert ''.join(shuffled) == shuffled_data == 'yhczewmkouqnaglvxrtsibjpdf'

    def test_skip(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')
        assert ''.join(sds.skip(10)) == 'klmnopqrstuvwxyz'

    def test_take(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')
        assert ''.join(sds.take(10)) == 'abcdefghij'

    def test_prefetch(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')

        # make sure that this placeholder method at least returns the exact same
        # dataset
        assert sds.prefetch() is sds

    def test_batch(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(4)

        assert isinstance(batched[0], SliceableDataset)
        assert ''.join(batched[0]) == 'abcd'
        assert [''.join(batch) for batch in batched] == [
            'abcd',
            'efgh',
            'ijkl',
            'mnop',
            'qrst',
            'uvwx',
            'yz',
        ]

    def test_unbatch(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(4)
        unbatched = batched.unbatch()

        assert ''.join(unbatched) == 'abcdefghijklmnopqrstuvwxyz'

    def test_flatten(self) -> None:
        sds = SliceableDataset('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(3)
        batched2 = batched.batch(2)
        batched3 = batched2.batch(5)
        batched3.flatten()

        # assert ''.join(flatten) == 'abcdefghijklmnopqrstuvwxyz'

    def test_element_spec(self) -> None:
        sds1 = SliceableDataset('abcd')
        assert sds1.element_spec == tf.TensorSpec(shape=(), dtype=tf.string)

        sds2 = SliceableDataset([0, 1, 2])
        assert sds2.element_spec == tf.TensorSpec(shape=(), dtype=tf.int32)

        sds3 = SliceableDataset([(0, 'a'), (1, 'b'), (2, 'c')])
        assert sds3.element_spec == (
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )

        sds4 = SliceableDataset([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        assert sds4.element_spec == (tf.TensorSpec(shape=(3), dtype=tf.int32))

        sds5 = SliceableDataset(
            [
                (
                    np.random.rand(3, 4),
                    {
                        'key1': [np.random.rand(2, 5), np.random.rand(2, 5), np.random.rand(2, 5)],
                        'key2': 'value1',
                        'key3': 10.5,
                    },
                    False,
                ),
                (
                    np.random.rand(3, 4),
                    {
                        'key1': [np.random.rand(2, 5), np.random.rand(2, 5)],
                        'key2': 'value2',
                        'key3': 3.14,
                    },
                    True,
                ),
            ]
        )
        assert sds5.element_spec == (
            tf.TensorSpec(shape=(3, 4), dtype=tf.float64),
            {
                'key1': tf.TensorSpec(shape=(3, 2, 5), dtype=tf.float64),
                'key2': tf.TensorSpec(shape=(), dtype=tf.string),
                'key3': tf.TensorSpec(shape=(), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(), dtype=tf.bool),
        )


def test_concatenate() -> None:
    sds1 = SliceableDataset('abcd')
    sds2 = SliceableDataset('efg')
    sds3 = SliceableDataset('hijkl')

    concat = concatenate((sds1, sds2, sds3))
    assert ''.join(concat) == 'abcdefghijkl'


def test_sliceable_to_tensorflow() -> None:
    sds = SliceableDataset(
        [np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4)]
    )
    ds = sliceable_dataset_to_tensorflow_dataset(sds)

    for sds_elem, ds_elem in zip(sds, ds.as_numpy_iterator()):
        assert np.allclose(sds_elem, ds_elem, atol=1e-8)
