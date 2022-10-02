from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.transforms import subset


def test_subset() -> None:
    ds_train = SliceableDataset.range(3)
    ds_val = SliceableDataset.range(3, 6)
    ds_test = SliceableDataset.range(6, 9)

    ds = DatasetTriplet(ds_train, ds_val, ds_test)

    train_subset = subset('train')(ds)
    val_subset = subset('val')(ds)
    test_subset = subset('test')(ds)

    assert list(train_subset) == list(ds_train)
    assert list(val_subset) == list(ds_val)
    assert list(test_subset) == list(ds_test)
