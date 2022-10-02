from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.management.allocators import json_describe
from boiling_learning.transforms import subset
from boiling_learning.utils.lazy import Lazy, LazyDescribed


def test_subset() -> None:
    ds_train = SliceableDataset.range(3)
    ds_val = SliceableDataset.range(3, 6)
    ds_test = SliceableDataset.range(6, 9)

    ds = DatasetTriplet(ds_train, ds_val, ds_test)

    train_subset = Lazy.from_value(ds) | subset('train')
    val_subset = Lazy.from_value(ds) | subset('val')
    test_subset = Lazy.from_value(ds) | subset('test')

    assert list(train_subset()) == list(ds_train) == [0, 1, 2]
    assert list(val_subset()) == list(ds_val) == [3, 4, 5]
    assert list(test_subset()) == list(ds_test) == [6, 7, 8]


def test_subset_lazy() -> None:
    ds_train = SliceableDataset.range(3)
    ds_val = SliceableDataset.range(3, 6)
    ds_test = SliceableDataset.range(6, 9)

    ds = LazyDescribed.from_value_and_description(
        DatasetTriplet(ds_train, ds_val, ds_test), 'my_dataset'
    )

    assert json_describe(ds) == 'my_dataset'

    train_subset = ds | subset('train')
    val_subset = ds | subset('val')
    test_subset = ds | subset('test')

    assert list(train_subset()) == list(ds_train) == [0, 1, 2]
    assert list(val_subset()) == list(ds_val) == [3, 4, 5]
    assert list(test_subset()) == list(ds_test) == [6, 7, 8]

    assert json_describe(train_subset) == {
        'contents': [
            'my_dataset',
            {
                'contents': {
                    'function': {
                        'contents': 'boiling_learning.transforms.subset',
                        'type': 'types.FunctionType',
                    },
                    'pack': {
                        'contents': {
                            'contents': [
                                {'contents': ['train'], 'type': 'builtins.tuple'},
                                {
                                    'contents': {'contents': {}, 'type': 'builtins.dict'},
                                    'type': 'boiling_learning.utils.frozendicts.frozendict',
                                },
                            ],
                            'type': 'builtins.tuple',
                        },
                        'type': 'boiling_learning.utils.functional.Pack',
                    },
                    'type': {
                        'contents': 'boiling_learning.preprocessing.transformers.Transformer',
                        'type': 'builtins.type',
                    },
                },
                'type': 'builtins.dict',
            },
        ],
        'type': 'builtins.tuple',
    }
    assert json_describe(val_subset) == {
        'contents': [
            'my_dataset',
            {
                'contents': {
                    'function': {
                        'contents': 'boiling_learning.transforms.subset',
                        'type': 'types.FunctionType',
                    },
                    'pack': {
                        'contents': {
                            'contents': [
                                {'contents': ['val'], 'type': 'builtins.tuple'},
                                {
                                    'contents': {'contents': {}, 'type': 'builtins.dict'},
                                    'type': 'boiling_learning.utils.frozendicts.frozendict',
                                },
                            ],
                            'type': 'builtins.tuple',
                        },
                        'type': 'boiling_learning.utils.functional.Pack',
                    },
                    'type': {
                        'contents': 'boiling_learning.preprocessing.transformers.Transformer',
                        'type': 'builtins.type',
                    },
                },
                'type': 'builtins.dict',
            },
        ],
        'type': 'builtins.tuple',
    }
    assert json_describe(test_subset) == {
        'contents': [
            'my_dataset',
            {
                'contents': {
                    'function': {
                        'contents': 'boiling_learning.transforms.subset',
                        'type': 'types.FunctionType',
                    },
                    'pack': {
                        'contents': {
                            'contents': [
                                {'contents': ['test'], 'type': 'builtins.tuple'},
                                {
                                    'contents': {'contents': {}, 'type': 'builtins.dict'},
                                    'type': 'boiling_learning.utils.frozendicts.frozendict',
                                },
                            ],
                            'type': 'builtins.tuple',
                        },
                        'type': 'boiling_learning.utils.functional.Pack',
                    },
                    'type': {
                        'contents': 'boiling_learning.preprocessing.transformers.Transformer',
                        'type': 'builtins.type',
                    },
                },
                'type': 'builtins.dict',
            },
        ],
        'type': 'builtins.tuple',
    }
