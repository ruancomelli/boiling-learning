from fractions import Fraction
from random import sample
from typing import Iterable, Optional

import numpy as np
import pytest

from boiling_learning.datasets.bridging import sliceable_dataset_to_tensorflow_dataset
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils.random import random_state


class MockDatabaseDataset(SliceableDataset[int]):
    def __init__(self) -> None:
        self.items = [index**2 for index in range(8)]
        self.database_fetches: list[list[int]] = []

    def __repr__(self) -> str:
        return f'MockDatabaseDataset({self.items})'

    def __len__(self) -> int:
        return len(self.items)

    def getitem_from_index(self, index: int) -> int:
        return self.items[index]

    def fetch(self, indices: Optional[Iterable[int]] = None) -> tuple[int, ...]:
        if indices is None:
            indices = range(len(self))

        indices = list(indices)
        self.database_fetches.append(indices)
        return tuple(self[index] for index in indices)


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
        sds = MockDatabaseDataset()

        assert isinstance(sds[1:3], SliceableDataset)

        assert list(sds[:]) == [0, 1, 4, 9, 16, 25, 36, 49]
        assert list(sds[:3]) == [0, 1, 4]
        assert list(sds[2:4]) == [4, 9]
        assert list(sds[3:]) == [9, 16, 25, 36, 49]
        assert list(sds[3:100]) == [9, 16, 25, 36, 49]

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

    def test_enumerate(self) -> None:
        sds = SliceableDataset.from_sequence('abcd')
        sds_enum = sds.enumerate()

        assert list(sds_enum) == [
            (0, 'a'),
            (1, 'b'),
            (2, 'c'),
            (3, 'd'),
        ]

    def test_extend(self) -> None:
        sds1 = SliceableDataset.from_sequence([4, 3, 2, 1])
        sds2 = SliceableDataset.from_sequence('abcd')
        sds = sds1.extend(sds2)

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

    def test_take(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        assert ''.join(sds.take(10)) == 'abcdefghij'
        assert ''.join(sds.take(Fraction(1, 3))) == 'abcdefgh'

    def test_skip(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        assert ''.join(sds.skip(10)) == 'klmnopqrstuvwxyz'
        assert ''.join(sds.skip(Fraction(1, 3))) == 'ijklmnopqrstuvwxyz'

    def test_prefetch(self) -> None:
        db = MockDatabaseDataset()

        prefetched = db.prefetch(3)
        it = iter(prefetched)

        assert not db.database_fetches

        assert next(it) == 0
        assert db.database_fetches == [[0, 1, 2]]
        assert next(it) == 1
        assert db.database_fetches == [[0, 1, 2]]
        assert next(it) == 4
        assert db.database_fetches == [[0, 1, 2]]
        assert next(it) == 9
        assert db.database_fetches == [[0, 1, 2], [3, 4, 5]]
        assert next(it) == 16
        assert db.database_fetches == [[0, 1, 2], [3, 4, 5]]
        assert next(it) == 25
        assert db.database_fetches == [[0, 1, 2], [3, 4, 5]]
        assert next(it) == 36
        assert db.database_fetches == [[0, 1, 2], [3, 4, 5], [6, 7]]
        assert next(it) == 49
        assert db.database_fetches == [[0, 1, 2], [3, 4, 5], [6, 7]]

        with pytest.raises(StopIteration):
            next(it)

        assert db.database_fetches == [[0, 1, 2], [3, 4, 5], [6, 7]]

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

        sds = SliceableDataset.from_sequence('abc')
        for batch in sds.batch(5):
            assert list(batch) == ['a', 'b', 'c']

    def test_unbatch(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(4)
        unbatched = batched.unbatch()

        assert ''.join(unbatched) == 'abcdefghijklmnopqrstuvwxyz'

    def test_map_batched(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')

        def _mapper(batch: SliceableDataset[str]) -> SliceableDataset[str]:
            return SliceableDataset.from_sequence(batch[::-1])

        mapped = sds.batch(4).map(_mapper).unbatch()
        assert ''.join(mapped) == 'dcbahgfelkjiponmtsrqxwvuzy'

    def test_flatten(self) -> None:
        sds = SliceableDataset.from_sequence('abcdefghijklmnopqrstuvwxyz')
        batched = sds.batch(3)
        batched2 = batched.batch(2)
        batched3 = batched2.batch(5)
        flatten = batched3.flatten()

        assert ''.join(flatten) == 'abcdefghijklmnopqrstuvwxyz'

    def test_repeat(self) -> None:
        db = MockDatabaseDataset()
        repeated = db.repeat(4)

        # fmt: off
        assert list(repeated) == [
            0, 1, 4, 9, 16, 25, 36, 49,
            0, 1, 4, 9, 16, 25, 36, 49,
            0, 1, 4, 9, 16, 25, 36, 49,
            0, 1, 4, 9, 16, 25, 36, 49
        ]
        # fmt: on

    def test_constantly(self) -> None:
        constant_dataset = SliceableDataset.constantly(3, count=4)

        assert list(constant_dataset) == [3, 3, 3, 3]
        assert constant_dataset[0] == 3
        assert constant_dataset[1] == 3
        assert constant_dataset[2] == 3
        assert constant_dataset[3] == 3
        assert constant_dataset.fetch() == (3, 3, 3, 3)


class TestComposedSliceableDataset:
    def test_fetch(self) -> None:
        db = MockDatabaseDataset()
        ds = db[[3, 1, 4, 1, 5]]
        assert ds.fetch() == (9, 1, 16, 1, 25)
        assert ds.fetch((2, 0, 0, 1)) == (16, 9, 9, 1)


def test_concatenate() -> None:
    sds1 = SliceableDataset.from_sequence('abcd')
    sds2 = SliceableDataset.from_sequence('efg')
    sds3 = SliceableDataset.from_sequence('hijkl')

    concat = SliceableDataset.concatenate(sds1, sds2, sds3)
    assert ''.join(concat) == 'abcdefghijkl'
    assert concat[0] == 'a'
    assert concat[5] == 'f'
    assert concat[11] == 'l'
    assert len(concat) == len(sds1) + len(sds2) + len(sds3)
    assert ''.join(concat[[0, 11, 5, 11]]) == 'alfl'
    assert ''.join(concat.fetch([0, 11, 5, 11])) == 'alfl'
    assert concat.fetch() == tuple('abcdefghijkl')


def test_sliceable_to_tensorflow() -> None:
    sds = SliceableDataset.from_sequence(
        [np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4)]
    )
    ds = sliceable_dataset_to_tensorflow_dataset(sds)

    for sds_elem, ds_elem in zip(sds, ds.as_numpy_iterator()):
        assert np.allclose(sds_elem, ds_elem, atol=1e-8)
