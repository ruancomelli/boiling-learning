from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import json_describe
from boiling_learning.transforms import subset


def test_subset() -> None:
    ds_train = SliceableDataset.range(3)
    ds_val = SliceableDataset.range(3, 6)
    ds_test = SliceableDataset.range(6, 9)

    ds = DatasetTriplet(ds_train, ds_val, ds_test)
    described_ds = LazyDescribed.from_value_and_description(ds, 'my_dataset')

    train_subset = described_ds | subset('train')
    val_subset = described_ds | subset('val')
    test_subset = described_ds | subset('test')

    assert list(train_subset()) == list(ds_train) == [0, 1, 2]
    assert list(val_subset()) == list(ds_val) == [3, 4, 5]
    assert list(test_subset()) == list(ds_test) == [6, 7, 8]


def test_subset_lazy() -> None:
    ds_train = SliceableDataset.range(3)
    ds_val = SliceableDataset.range(3, 6)
    ds_test = SliceableDataset.range(6, 9)

    ds = LazyDescribed.from_value_and_description(
        DatasetTriplet(ds_train, ds_val, ds_test),
        'my_dataset',
    )

    assert json_describe(ds) == 'my_dataset'

    train_subset = ds | subset('train')
    val_subset = ds | subset('val')
    test_subset = ds | subset('test')

    assert list(train_subset()) == list(ds_train) == [0, 1, 2]
    assert list(val_subset()) == list(ds_val) == [3, 4, 5]
    assert list(test_subset()) == list(ds_test) == [6, 7, 8]

    assert json_describe(train_subset) == [
        'builtins.tuple',
        'my_dataset',
        {
            'function': ['types.FunctionType', 'boiling_learning.transforms.subset'],
            'pack': ['boiling_learning.utils.functional.Pack', ['train'], {}],
            'type': [
                'builtins.type',
                'boiling_learning.preprocessing.transformers.Transformer',
            ],
        },
    ]
    assert json_describe(val_subset) == [
        'builtins.tuple',
        'my_dataset',
        {
            'function': ['types.FunctionType', 'boiling_learning.transforms.subset'],
            'pack': ['boiling_learning.utils.functional.Pack', ['val'], {}],
            'type': [
                'builtins.type',
                'boiling_learning.preprocessing.transformers.Transformer',
            ],
        },
    ]
    assert json_describe(test_subset) == [
        'builtins.tuple',
        'my_dataset',
        {
            'function': ['types.FunctionType', 'boiling_learning.transforms.subset'],
            'pack': ['boiling_learning.utils.functional.Pack', ['test'], {}],
            'type': [
                'builtins.type',
                'boiling_learning.preprocessing.transformers.Transformer',
            ],
        },
    ]
