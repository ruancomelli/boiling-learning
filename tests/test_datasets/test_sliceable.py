from boiling_learning.datasets.sliceable import SliceableDataset


def test_basics() -> None:
    data = [0, 10, 200, 3, 45]

    sds = SliceableDataset(data)

    assert len(sds) == len(data) == 5
    assert list(sds) == data
    assert sds[2] == 200
    assert sds[0] == 0
    assert sds[-1] == 45


def test_slicing() -> None:
    sds = SliceableDataset([0, 10, 200, 3, 45])

    assert isinstance(sds[1:3], SliceableDataset)

    assert list(sds[:]) == [0, 10, 200, 3, 45]
    assert list(sds[:3]) == [0, 10, 200]
    assert list(sds[2:4]) == [200, 3]
    assert list(sds[3:]) == [3, 45]


def test_masking() -> None:
    sds = SliceableDataset([0, 10, 200, 3, 45])

    assert isinstance(sds[[False, True, False, False, True]], SliceableDataset)
    assert list(sds[[False, False, False, False, False]]) == []
    assert list(sds[[False, True, False, False, True]]) == [10, 45]
    assert list(sds[[True, True, True, True, True]]) == [0, 10, 200, 3, 45]


def test_selecting() -> None:
    sds = SliceableDataset([0, 10, 200, 3, 45])

    assert isinstance(sds[[0, 3, 3, -1, 2, -1]], SliceableDataset)
    assert list(sds[[2, 2, 0, -1, 2, -1]]) == [200, 200, 0, 45, 200, 45]
    assert list(sds[[]]) == []


def test_zip() -> None:
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


def test_apply() -> None:
    def stringify(sds: SliceableDataset[int]) -> SliceableDataset[str]:
        return SliceableDataset([str(elem) for elem in sds])

    sds = SliceableDataset([3, 1, 4, 1, 5])
    assert list(sds.apply(stringify)) == list(stringify(sds)) == ['3', '1', '4', '1', '5']


def test_concatenate() -> None:
    sds1 = SliceableDataset([4, 3, 2, 1])
    sds2 = SliceableDataset('abcd')
    sds = sds1.concatenate(sds2)

    assert isinstance(sds, SliceableDataset)
    assert list(sds) == [4, 3, 2, 1, 'a', 'b', 'c', 'd']


def test_enumerate() -> None:
    sds = SliceableDataset('abcd')

    assert isinstance(sds.enumerate(), SliceableDataset)
    assert list(sds.enumerate()) == [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]


def test_map() -> None:
    sds = SliceableDataset('abcd')
    assert list(sds.map(str.upper)) == ['A', 'B', 'C', 'D']
