from boiling_learning.datasets.sliceable import (
    SliceableDataset,
    load_sliceable_dataset,
    save_sliceable_dataset,
)
from boiling_learning.utils.utils import tempfilepath


def test_io() -> None:
    ds = SliceableDataset.zip(SliceableDataset.range(2, 6), SliceableDataset(['a', 'b', 'c', 'd']))
    original_data = list(ds)

    with tempfilepath() as path:
        save_sliceable_dataset(ds, path)
        recovered = load_sliceable_dataset(path)
        recovered_data = list(recovered)

    assert recovered_data == [(2, 'a'), (3, 'b'), (4, 'c'), (5, 'd')]
    assert original_data == recovered_data
