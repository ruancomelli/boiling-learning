import enum
import functools
from collections import deque
from fractions import Fraction
from typing import Callable, Iterable, Optional, Type, TypeVar, Union

import funcy
import more_itertools as mit
import tensorflow as tf
from dataclassy import dataclass
from frozendict import frozendict
from tensorflow.data import AUTOTUNE

import boiling_learning.utils.mathutils as mathutils
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.preprocessing.transformers import Transformer

_sentinel = object()
_T = TypeVar('_T')


@dataclass(kwargs=True)
class DatasetSplits:
    train: Optional[Fraction] = None
    test: Optional[Fraction] = None
    val: Optional[Fraction] = Fraction(0)

    def __post_init__(self) -> None:
        splits = (self.train, self.val, self.test)
        n_nones = splits.count(None)
        if n_nones > 1:
            raise ValueError(
                'at most one of *train*, *val* and *test* can be inferred (by passing None)'
            )

        if n_nones == 1:
            names = ('train', 'val', 'test')
            dct = funcy.zipdict(names, splits)
            for name, split in dct.items():
                if split is None:
                    others = funcy.omit(dct, [name])
                    others_sum = sum(others.values())

                    if not (0 < others_sum <= 1):
                        raise ValueError(
                            'it is required that 0 < '
                            + ' + '.join(
                                f'*{other}*' for other in others.keys()
                            )
                            + ' <= 1'
                        )

                    split = 1 - others_sum
                    object.__setattr__(self, name, split)
                    splits = (self.train, self.val, self.test)
                    break

        if sum(splits) != 1:
            raise ValueError('*train* + *val* + *test* must equal 1')

        if not (
            0 < self.train < 1 and 0 <= self.val < 1 and 0 < self.test < 1
        ):
            raise ValueError(
                'it is required that 0 < (*train*, *test*) < 1 and 0 <= *val* < 1'
            )


class Split(enum.Flag):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()

    @classmethod
    def get_split(cls: Type[_T], s: str, default=_sentinel) -> _T:
        if s in cls:
            return s
        else:
            return cls.from_string(s, default=default)

    @classmethod
    def from_string(cls: Type[_T], s: str, default=_sentinel) -> _T:
        if default is not _sentinel:
            return cls.FROM_STR_TABLE.get(s, default)

        try:
            return cls.FROM_STR_TABLE[s]
        except KeyError:
            raise KeyError(
                f'string {s} was not found in the conversion table.'
                f'Available values are {tuple(cls.FROM_STR_TABLE.keys())}.'
            )

    def to_str(self):
        return self.name.lower()


Split.FROM_STR_TABLE = frozendict(
    {
        key_str: split_subset
        for keys, split_subset in (
            (('train',), Split.TRAIN),
            (('val', 'validation'), Split.VAL),
            (
                tuple(
                    connector.join(('train', validation_key))
                    for connector in ('_', '_and_')
                    for validation_key in ('val', 'validation')
                ),
                Split.TRAIN | Split.VAL,
            ),
            (('test',), Split.TEST),
            (('all',), Split.TRAIN | Split.VAL | Split.TEST),
        )
        for key_str in keys
    }
)


def concatenate(datasets: Iterable[tf.data.Dataset]) -> tf.data.Dataset:
    datasets = deque(datasets)

    if not datasets:
        raise ValueError('argument *datasets* must be a non-empty iterable.')

    ds = datasets.popleft()
    for dataset in datasets:
        ds = ds.concatenate(dataset)

    return ds


def flatten_zip(*ds: tf.data.Dataset) -> tf.data.Dataset:
    if len(ds) == 1:
        return ds[0]
    else:
        return tf.data.Dataset.zip(ds)


def train_val_test_split(
    ds: tf.data.Dataset, splits: DatasetSplits, shuffle: bool = False
) -> DatasetTriplet:
    """#TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    This function is deterministic in the sense that it outputs the same splits given the same inputs.
    Consequently it is safe to be used early in the pipeline to avoid consuming test data.
    """

    if splits.val == 0:
        split_train, split_test = mathutils.proportional_ints(
            splits.train, splits.test
        )
        window_shift = split_train + split_test
        if shuffle:
            ds = ds.shuffle(window_shift, reshuffle_each_iteration=False)
        ds_train = ds.window(split_train, window_shift).flat_map(flatten_zip)
        ds_test = (
            ds.skip(split_train)
            .window(split_test, window_shift)
            .flat_map(flatten_zip)
        )
        ds_val = None
    else:
        split_train, split_val, split_test = mathutils.proportional_ints(
            splits.train, splits.val, splits.test
        )
        window_shift = split_train + split_val + split_test
        if shuffle:
            ds = ds.shuffle(window_shift, reshuffle_each_iteration=False)
        ds_train = ds.window(split_train, window_shift).flat_map(flatten_zip)
        ds_val = (
            ds.skip(split_train)
            .window(split_val, window_shift)
            .flat_map(flatten_zip)
        )
        ds_test = (
            ds.skip(split_train + split_val)
            .window(split_test, window_shift)
            .flat_map(flatten_zip)
        )

    return ds_train, ds_val, ds_test


def train_val_test_split_concat(
    datasets: Iterable[tf.data.Dataset], splits: DatasetSplits
) -> DatasetTriplet:
    """#TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    The paramenter *splits* is must contain only integers, with the possible exception of *val*, which can be *None*.

    This function is deterministic in the sense that it outputs the same splits given the same inputs.
    Consequently it is safe to be used early in the pipeline to avoid consuming test data.
    """
    datasets = deque(datasets)

    if not datasets:
        raise ValueError('argument *datasets* must be a non-empty iterable.')

    ds = datasets.popleft()
    ds_train, ds_val, ds_test = train_val_test_split(ds, splits)

    if ds_val is None:
        for ds in datasets:
            ds_train_, _, ds_test_ = train_val_test_split(ds, splits)
            ds_train = ds_train.concatenate(ds_train_)
            ds_test = ds_test.concatenate(ds_test_)
    else:
        for ds in datasets:
            ds_train_, ds_val_, ds_test_ = train_val_test_split(ds, splits)
            ds_train = ds_train.concatenate(ds_train_)
            ds_val = ds_val.concatenate(ds_val_)
            ds_test = ds_test.concatenate(ds_test_)

    return ds_train, ds_val, ds_test


def bulk_split(
    ds: tf.data.Dataset, splits: DatasetSplits, length: Optional[int] = None
) -> DatasetTriplet:
    if length is None:
        length = calculate_dataset_size(ds)

    train_len = int(splits.train * length)
    val_len = int(splits.val * length)
    test_len = int(splits.test * length)

    ds_train = ds.take(train_len)
    ds_val = ds.skip(train_len).take(val_len) if val_len > 0 else None
    ds_test = ds.skip(train_len + val_len).take(test_len)

    return ds_train, ds_val, ds_test


def take(
    ds: tf.data.Dataset,
    count: Optional[Union[int, Fraction]],
    unbatch_dim: Optional[int] = None,
    unbatch_key: Optional[int] = None,
) -> tf.data.Dataset:
    if count is None:
        return ds

    unbatch = unbatch_dim is not None

    if unbatch_key is not None and not unbatch:
        raise ValueError(
            '*unbatch_key* must be *None* if not unbatching.'
            ' Unbatching happens iff *unbatch_dim* is not *None*.'
        )

    if unbatch:
        return apply_unbatched(
            ds,
            functools.partial(
                take, count=count, unbatch_dim=None, unbatch_key=None
            ),
            dim=unbatch_dim,
            key=unbatch_key,
        )

    if isinstance(count, int):
        return ds.take(count)
    elif isinstance(count, Fraction):
        return train_val_test_split(ds, splits=DatasetSplits(train=count))[0]
    else:
        raise TypeError(
            f'*count* must be either *int* or *Fraction*, got {type(count)}.'
        )


def calculate_batch_size(
    dataset: tf.data.Dataset, dim: int = 0, key: Optional[int] = None
) -> int:
    elem = mit.first(dataset)
    if key is not None:
        elem = elem[key]
    return elem.shape[dim]


def calculate_dataset_size(
    dataset: tf.data.Dataset, batched_dim: Optional[int] = None
) -> int:
    if batched_dim is not None:
        batch_size_calculator = functools.partial(
            calculate_batch_size, dim=batched_dim
        )
        return sum(map(batch_size_calculator, dataset))
    else:
        try:
            return len(dataset)
        except TypeError:
            return mit.ilen(dataset)


def apply_unbatched(
    dataset: tf.data.Dataset,
    apply: Callable[[tf.data.Dataset], tf.data.Dataset],
    dim: int = 0,
    key: Optional[int] = None,
) -> tf.data.Dataset:
    batch_size = calculate_batch_size(dataset, dim=dim, key=key)
    return dataset.unbatch().apply(apply).batch(batch_size)


def map_unbatched(
    dataset: tf.data.Dataset,
    map_fn: Callable,
    dim: int = 0,
    key: Optional[int] = None,
) -> tf.data.Dataset:
    return apply_unbatched(
        dataset, lambda ds: ds.map(map_fn), dim=dim, key=key
    )


def filter_unbatched(
    dataset: tf.data.Dataset,
    pred_fn: Callable[..., bool],
    dim: int = 0,
    key: Optional[int] = None,
) -> tf.data.Dataset:
    return apply_unbatched(
        dataset, lambda ds: ds.filter(pred_fn), dim=dim, key=key
    )


def apply_transformers(
    ds: tf.data.Dataset, transformers: Iterable[Transformer]
) -> tf.data.Dataset:
    for transformer in transformers:
        ds = ds.map(
            transformer.as_tf_py_function(pack_tuple=True),
            num_parallel_calls=AUTOTUNE,
        )
    return ds
