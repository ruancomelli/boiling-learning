from fractions import Fraction
from random import sample
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from boiling_learning.datasets.bridging import sliceable_dataset_to_tensorflow_dataset
from boiling_learning.datasets.sliceable import SliceableDataset, concatenate
from boiling_learning.utils.random import random_state


class TestSliceableDataset:
    def test_basics(self) -> None:
        data = [0, 10, 200, 3, 45]

        sds = SliceableDataset.from_sequence(data)

        assert len(sds) == len(data) == 5
        assert list(sds) == data
        assert sds[2] == 200
        assert sds[0] == 0
        assert sds[-1] == 45

    def test_slicing(self) -> None:
        sds = SliceableDataset.from_sequence([0, 10, 200, 3, 45])

        assert isinstance(sds[1:3], SliceableDataset)

        assert list(sds[:]) == [0, 10, 200, 3, 45]
        assert list(sds[:3]) == [0, 10, 200]
        assert list(sds[2:4]) == [200, 3]
        assert list(sds[3:]) == [3, 45]

    def test_masking(self) -> None:
        sds = SliceableDataset.from_sequence([0, 10, 200, 3, 45])

        assert isinstance(sds[[False, True, False, False, True]], SliceableDataset)
        assert not list(sds[[False, False, False, False, False]])
        assert list(sds[[False, True, False, False, True]]) == [10, 45]
        assert list(sds[[True, True, True, True, True]]) == [0, 10, 200, 3, 45]

    def test_selecting(self) -> None:
        sds = SliceableDataset.from_sequence([0, 10, 200, 3, 45])

        assert isinstance(sds[[0, 3, 3, -1, 2, -1]], SliceableDataset)
        assert list(sds[[2, 2, 0, -1, 2, -1]]) == [200, 200, 0, 45, 200, 45]
        assert not list(sds[[]])

    def test_zip(self) -> None:
        sds1 = SliceableDataset.from_sequence([10, 5, 2, 8])
        sds2 = SliceableDataset.from_sequence('abcd')
        sds3 = SliceableDataset.from_sequence(range(4))

        sds = SliceableDataset.zip(sds1, sds2, sds3)
        assert sds[0] == (10, 'a', 0)
        assert list(sds) == [
            (10, 'a', 0),
            (5, 'b', 1),
            (2, 'c', 2),
            (8, 'd', 3),
        ]

    def test_concatenate(self) -> None:
        sds1 = SliceableDataset.from_sequence([4, 3, 2, 1])
        sds2 = SliceableDataset.from_sequence('abcd')
        sds = sds1.concatenate(sds2)

        assert isinstance(sds, SliceableDataset)
        assert list(sds) == [4, 3, 2, 1, 'a', 'b', 'c', 'd']

    def test_map(self) -> None:
        sds = SliceableDataset.from_sequence('abcd')
        assert list(sds.map(str.upper)) == ['A', 'B', 'C', 'D']

    def test_split(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        splits = sds.split(5, 0, None, Fraction(1, 4))

        assert ''.join(splits[0]) == 'abcde'
        assert not ''.join(splits[1])
        assert ''.join(splits[2]) == 'fghijklmnopqrst'
        assert ''.join(splits[3]) == 'uvwxyz'

    def test_shuffle(self) -> None:
        data = 'abcdefghijklmnopqrstuvwxyz'
        with random_state(1997):
            shuffled_data = ''.join(sample(data, k=len(data)))

        sds = SliceableDataset.from_sequence(data)
        with random_state(1997):
            shuffled = sds.shuffle()

        assert ''.join(shuffled) == shuffled_data == 'yhczewmkouqnaglvxrtsibjpdf'
        assert ''.join(shuffled.fetch(range(5, 10))) == 'wmkou'

    def test_skip(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        assert ''.join(sds.skip(10)) == 'klmnopqrstuvwxyz'

    def test_take(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        assert ''.join(sds.take(10)) == 'abcdefghij'

    def test_prefetch(self) -> None:
        database_fetches: List[List[int]] = []

        class MockDatabaseDataset(SliceableDataset[int]):
            def __len__(self) -> int:
                return 20

            def getitem_from_index(self, index: int) -> int:
                return index ** 2

            def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[int, ...]:
                if indices is None:
                    indices = range(len(self))

                indices = list(indices)
                database_fetches.append(indices)
                return tuple(self[indices])

        prefetched = MockDatabaseDataset().prefetch(3)
        it = iter(prefetched)

        assert not database_fetches

        assert next(it) == 0
        assert database_fetches == [[0, 1, 2]]
        assert next(it) == 1
        assert database_fetches == [[0, 1, 2]]
        assert next(it) == 4
        assert database_fetches == [[0, 1, 2]]
        assert next(it) == 9
        assert database_fetches == [[0, 1, 2], [3, 4, 5]]
        assert next(it) == 16
        assert database_fetches == [[0, 1, 2], [3, 4, 5]]
        assert next(it) == 25
        assert database_fetches == [[0, 1, 2], [3, 4, 5]]
        assert next(it) == 36
        assert database_fetches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        assert next(it) == 49
        assert database_fetches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        assert next(it) == 64
        assert database_fetches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_batch(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
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
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(4)
        unbatched = batched.unbatch()

        assert ''.join(unbatched) == 'abcdefghijklmnopqrstuvwxyz'

    def test_flatten(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(3)
        batched2 = batched.batch(2)
        batched3 = batched2.batch(5)
        batched3.flatten()

        # assert ''.join(flatten) == 'abcdefghijklmnopqrstuvwxyz'

    def test_element_spec(self) -> None:
        sds1 = SliceableDataset.from_sequence('abcd')
        assert sds1.element_spec == tf.TensorSpec(shape=(), dtype=tf.string)

        sds2 = SliceableDataset.from_sequence([0, 1, 2])
        assert sds2.element_spec == tf.TensorSpec(shape=(), dtype=tf.int32)

        sds3 = SliceableDataset.from_sequence([(0, 'a'), (1, 'b'), (2, 'c')])
        assert sds3.element_spec == (
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )

        sds4 = SliceableDataset.from_sequence([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        assert sds4.element_spec == (tf.TensorSpec(shape=(3), dtype=tf.int32))

        sds5 = SliceableDataset.from_sequence(
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
    sds1 = SliceableDataset.from_sequence('abcd')
    sds2 = SliceableDataset.from_sequence('efg')
    sds3 = SliceableDataset.from_sequence('hijkl')

    concat = concatenate((sds1, sds2, sds3))
    assert ''.join(concat) == 'abcdefghijkl'
    assert concat[0] == 'a'
    assert concat[5] == 'f'
    assert concat[11] == 'l'
    assert len(concat) == len(sds1) + len(sds2) + len(sds3)


def test_sliceable_to_tensorflow() -> None:
    sds = SliceableDataset.from_sequence(
        [np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4)]
    )
    ds = sliceable_dataset_to_tensorflow_dataset(sds)

    for sds_elem, ds_elem in zip(sds, ds.as_numpy_iterator()):
        assert np.allclose(sds_elem, ds_elem, atol=1e-8)
