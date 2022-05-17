from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Optional, Tuple, TypeVar, Union

import funcy
import tensorflow as tf
from decorator import decorator
from funcy import rpartial

from boiling_learning.io.storage import Metadata, deserialize, load, save, serialize
from boiling_learning.utils import mathutils, resolve
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.sentinels import EMPTY

_T = TypeVar('_T')


class DatasetTriplet(Tuple[_T, Optional[_T], _T], Generic[_T]):
    pass


@dataclass(frozen=True)
class DatasetSplits:
    train: Optional[Fraction] = None
    test: Optional[Fraction] = None
    val: Optional[Fraction] = Fraction(0)

    def __post_init__(self) -> None:
        splits = (self.train, self.val, self.test)
        n_nones = splits.count(None)
        if n_nones > 1:
            raise ValueError(
                'at most one of *train*, *val* and *test* can be inferred (by passing `None`)'
            )

        if n_nones == 1:
            names = ('train', 'val', 'test')
            dct = funcy.zipdict(names, splits)
            for name, split in dct.items():
                if split is None:
                    others = funcy.omit(dct, [name])
                    others_sum = sum(others.values())

                    if not 0 < others_sum <= 1:
                        raise ValueError(
                            'it is required that 0 < '
                            + ' + '.join(f'*{other}*' for other in others.keys())
                            + ' <= 1'
                        )

                    split = 1 - others_sum
                    object.__setattr__(self, name, split)
                    splits = (self.train, self.val, self.test)
                    break

        if sum(splits) != 1:
            raise ValueError('*train* + *val* + *test* must equal 1')

        if not (0 < self.train < 1 and 0 <= self.val < 1 and 0 < self.test < 1):
            raise ValueError('it is required that 0 < (*train*, *test*) < 1 and 0 <= *val* < 1')


@decorator
def none_aware(
    f: Callable[..., _T],
    ds: Optional[tf.data.Dataset],
    *args: Any,
    **kwargs: Any,
) -> Optional[_T]:
    '''
    Takes a function which requires a tf.data.Dataset as first argument
    and outputs another function which also accepts *None* instead of a
    dataset, making it aware of *None* datasets. In this case, *None* is
    silently returned without errors.

    For example:
    >>> ds = tf.data.Dataset.range(10)
    >>> def take(ds: tf.data.Dataset, count: int) -> tf.data.Dataset:
    ...     return ds.take(count)
    >>> list(take(ds, 2).as_numpy_iterator())
    [0, 1]
    >>> ds_val = take(None, 2)
    Traceback (most recent call last):
     ...
    AttributeError: 'NoneType' object has no attribute 'take'
    >>> safe_take = none_aware(take)
    >>> safe_take(None)
    None
    '''
    return f(ds, *args, **kwargs) if ds is not None else None


def is_dataset_triplet(obj: Any) -> bool:
    return isinstance(obj, tuple) and len(obj) == 3


@decorator
def triplet_aware(
    f: Callable[..., _T],
    ds: Union[tf.data.Dataset, DatasetTriplet],
    *args: Any,
    **kwargs: Any,
) -> Union[_T, Tuple[_T, _T, _T]]:
    '''
    Takes a function which requires a tf.data.Dataset as first argument
    and outputs another function which also accepts a `DatasetTriplet` instead
    of a dataset, making it aware of dataset triplets. In this case, the
    original function is applied once for each subset (train/val/test) and the
    three return values are returned as a tuple.

    For example:
    >>> ds_train = tf.data.Dataset.range(60)
    >>> ds_val = tf.data.Dataset.range(60, 80)
    >>> ds_test = tf.data.Dataset.range(80, 100)
    >>> @triplet_aware
    ... def take(ds: tf.data.Dataset, count: int) -> tf.data.Dataset:
    ...     return ds.take(count)
    >>> list(take(ds_train, 2).as_numpy_iterator())
    [0, 1]
    >>> ds_train, ds_val, ds_test = take((ds_train, ds_val, ds_test), 3)
    >>> list(ds_train.as_numpy_iterator())
    [0, 1, 2]
    >>> list(ds_val.as_numpy_iterator())
    [60, 61, 62]
    >>> list(ds_test.as_numpy_iterator())
    [80, 81, 82]
    '''
    partialized: Callable[[tf.data.Dataset], _T] = rpartial(f, *args, **kwargs)

    if not is_dataset_triplet(ds):
        return partialized(ds)

    ds_train, ds_val, ds_test = ds
    return partialized(ds_train), partialized(ds_val), partialized(ds_test)


def concatenate(datasets: Iterable[tf.data.Dataset]) -> tf.data.Dataset:
    datasets = deque(datasets)

    if not datasets:
        raise ValueError('argument *datasets* must be a non-empty iterable.')

    ds = datasets.popleft()
    for dataset in datasets:
        ds = ds.concatenate(dataset)

    return ds


def flatten_zip(*ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds)


def train_val_test_split(
    ds: tf.data.Dataset, splits: DatasetSplits, shuffle: bool = False
) -> DatasetTriplet:
    """#TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    This function is deterministic in the sense that it outputs the same splits given the same
    inputs.
    Consequently it is safe to be used early in the pipeline to avoid consuming test data.
    """

    if splits.val == 0:
        split_train, split_test = mathutils.proportional_ints(splits.train, splits.test)
        window_shift = split_train + split_test
        if shuffle:
            ds = ds.shuffle(window_shift, reshuffle_each_iteration=False)
        ds_train = ds.window(split_train, window_shift).flat_map(flatten_zip)
        ds_test = ds.skip(split_train).window(split_test, window_shift).flat_map(flatten_zip)
        ds_val = None
    else:
        split_train, split_val, split_test = mathutils.proportional_ints(
            splits.train, splits.val, splits.test
        )
        window_shift = split_train + split_val + split_test
        if shuffle:
            ds = ds.shuffle(window_shift, reshuffle_each_iteration=False)
        ds_train = ds.window(split_train, window_shift).flat_map(flatten_zip)
        ds_val = ds.skip(split_train).window(split_val, window_shift).flat_map(flatten_zip)
        ds_test = (
            ds.skip(split_train + split_val).window(split_test, window_shift).flat_map(flatten_zip)
        )

    return ds_train, ds_val, ds_test


@triplet_aware
@none_aware
def take(ds: tf.data.Dataset, count: Optional[Union[int, Fraction]]) -> tf.data.Dataset:
    if count is None:
        return ds

    if isinstance(count, int):
        return ds.take(count)

    if isinstance(count, Fraction):
        return train_val_test_split(ds, splits=DatasetSplits(train=count))[0]

    raise TypeError(f'*count* must be either an *int* or a *Fraction*, got {type(count)}.')


@triplet_aware
@none_aware
def features(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(lambda x, y: x)


@triplet_aware
@none_aware
def targets(ds: tf.data.Dataset, key: Any = EMPTY) -> tf.data.Dataset:
    return ds.map(lambda x, y: y) if key is EMPTY else ds.map(lambda x, y: y[key])


@serialize.instance(DatasetTriplet)
def _serialize_dataset_triplet(instance: DatasetTriplet[Any], path: Path) -> None:
    path = resolve(path, dir=True)

    ds_train, ds_val, ds_test = instance

    save(ds_train, path / 'train')
    save(ds_val, path / 'val')
    save(ds_test, path / 'test')


@deserialize.dispatch(DatasetTriplet)
def _deserialize_dataset_triplet(path: Path, metadata: Metadata) -> DatasetTriplet[Any]:
    ds_train = load(path / 'train')
    ds_val = load(path / 'val')
    ds_test = load(path / 'test')

    return DatasetTriplet(ds_train, ds_val, ds_test)
