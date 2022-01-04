import random
from collections import deque
from fractions import Fraction
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import funcy
import more_itertools as mit
import numpy as np
import tensorflow as tf
from dataclassy import dataclass
from decorator import decorator
from funcy.funcs import rpartial
from tensorflow.data import AUTOTUNE

import boiling_learning.utils.mathutils as mathutils
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.utils.dtypes import auto_spec
from boiling_learning.utils.iterutils import distance_maximized_evenly_spaced_indices
from boiling_learning.utils.sentinels import EMPTY
from boiling_learning.utils.slicerators import Slicerator

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
                'at most one of *train*, *val* and *test* can be inferred (by passing `None`)'
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


def split_sizes(splits: DatasetSplits, total_length: int) -> Tuple[int, int, int]:
    return (
        int(splits.train * total_length),
        int(splits.val * total_length),
        int(splits.test * total_length),
    )


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
    if ds is None:
        return None
    else:
        return f(ds, *args, **kwargs)


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
        return bulk_split(ds, splits, length=calculate_dataset_size(ds))

    train_len = int(splits.train * length)
    val_len = int(splits.val * length)
    test_len = int(splits.test * length)

    ds_train = ds.take(train_len)
    ds_val = ds.skip(train_len).take(val_len) if val_len > 0 else None
    ds_test = ds.skip(train_len + val_len).take(test_len)

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
def calculate_batch_size(dataset: tf.data.Dataset, dim: int = 0, key: Optional[int] = None) -> int:
    elem = mit.first(dataset)
    if key is not None:
        elem = elem[key]
    return elem.shape[dim]


@triplet_aware
@none_aware
def calculate_dataset_size(dataset: tf.data.Dataset, batched_dim: Optional[int] = None) -> int:
    if batched_dim is not None:
        return sum(calculate_batch_size(batch, dim=batched_dim) for batch in dataset)

    try:
        return len(dataset)
    except TypeError:
        return mit.ilen(dataset)


@triplet_aware
@none_aware
def apply_unbatched(
    dataset: tf.data.Dataset,
    apply: Callable[[tf.data.Dataset], tf.data.Dataset],
    dim: int = 0,
    key: Optional[int] = None,
) -> tf.data.Dataset:
    batch_size = calculate_batch_size(dataset, dim=dim, key=key)
    return dataset.unbatch().apply(apply).batch(batch_size)


@triplet_aware
@none_aware
def apply_flattened(
    dataset: tf.data.Dataset,
    apply: Callable[[tf.data.Dataset], tf.data.Dataset],
) -> tf.data.Dataset:
    return apply_unbatched(dataset, apply, dim=0, key=0)


@triplet_aware
@none_aware
def map_unbatched(
    dataset: tf.data.Dataset,
    map_fn: Callable,
    dim: int = 0,
    key: Optional[int] = None,
) -> tf.data.Dataset:
    return apply_unbatched(
        dataset, lambda ds: ds.map(map_fn, num_parallel_calls=AUTOTUNE), dim=dim, key=key
    )


@triplet_aware
@none_aware
def map_flattened(dataset: tf.data.Dataset, map_fn: Callable) -> tf.data.Dataset:
    return map_unbatched(dataset, map_fn, dim=0, key=0)


@triplet_aware
@none_aware
def filter_unbatched(
    dataset: tf.data.Dataset,
    pred_fn: Callable[..., bool],
    dim: int = 0,
    key: Optional[int] = None,
) -> tf.data.Dataset:
    return apply_unbatched(dataset, lambda ds: ds.filter(pred_fn), dim=dim, key=key)


@triplet_aware
@none_aware
def filter_flattened(
    dataset: tf.data.Dataset,
    pred_fn: Callable[..., bool],
) -> tf.data.Dataset:
    return filter_unbatched(dataset, pred_fn, dim=0, key=0)


@triplet_aware
@none_aware
def apply_transformers(
    ds: tf.data.Dataset, transformers: Iterable[Transformer]
) -> tf.data.Dataset:
    for transformer in transformers:
        ds = ds.map(
            transformer.as_tf_py_function(pack_tuple=True),
            num_parallel_calls=AUTOTUNE,
        )
    return ds


@triplet_aware
@none_aware
def features(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(lambda x, y: x)


@triplet_aware
@none_aware
def targets(ds: tf.data.Dataset, key: Any = EMPTY) -> tf.data.Dataset:
    if key is EMPTY:
        return ds.map(lambda x, y: y)
    else:
        return ds.map(lambda x, y: y[key])


def preprocess_slicerator(
    slicerator: Slicerator[_T],
    *,
    take: Optional[Union[int, Fraction]] = None,
    shuffle: bool = False,
) -> Slicerator[_T]:
    total: int = len(slicerator)

    if isinstance(take, Fraction):
        take = int(take * total)

    if isinstance(take, int):
        keep_indices = distance_maximized_evenly_spaced_indices(total=total, count=take)
        slicerator = slicerator[keep_indices]

    if shuffle:
        indices = range(total)
        shuffled_indices = random.sample(indices, k=total)
        slicerator = slicerator[shuffled_indices]

    return slicerator


def _slicerator_to_dataset(slicerator: Slicerator[Any]) -> tf.data.Dataset:
    sample = slicerator[0]
    typespec = auto_spec(sample)

    return tf.data.Dataset.from_generator(lambda: slicerator, output_signature=typespec)


def slicerator_to_dataset(
    slicerator: Slicerator[Any],
    *,
    dataset_size: Optional[Union[int, Fraction]] = None,
    shuffle: bool = False,
) -> tf.data.Dataset:
    slicerator = preprocess_slicerator(slicerator, take=dataset_size, shuffle=shuffle)
    return _slicerator_to_dataset(slicerator)


def slicerator_to_dataset_triplet(
    slicerator: Slicerator[Any],
    splits: DatasetSplits,
    *,
    dataset_size: Optional[Union[int, Fraction]] = None,
    shuffle: bool = False,
) -> DatasetTriplet:
    train_size, val_size, _ = split_sizes(splits, len(slicerator))

    train_set = slicerator[:train_size]
    val_set = slicerator[train_size : train_size + val_size] if val_size > 0 else None
    test_set = slicerator[train_size + val_size :]

    _slicerator_to_dataset = partial(
        slicerator_to_dataset, dataset_size=dataset_size, shuffle=shuffle
    )

    return (
        _slicerator_to_dataset(train_set),
        _slicerator_to_dataset(val_set) if val_set is not None else None,
        _slicerator_to_dataset(test_set),
    )


def experiment_video_to_dataset_triplet(
    ev: ExperimentVideo,
    splits: DatasetSplits,
    *,
    image_preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    select_columns: Optional[Union[str, List[str]]] = None,
    dataset_size: Optional[Union[int, Fraction]] = None,
    shuffle: bool = False,
) -> DatasetTriplet:
    pairs = ev.as_pairs(image_preprocessor=image_preprocessor, select_columns=select_columns)

    return slicerator_to_dataset_triplet(pairs, splits, dataset_size=dataset_size, shuffle=shuffle)


def experiment_video_to_sequential_dataset_triplet(
    ev: ExperimentVideo,
    splits: DatasetSplits,
    select_columns: Optional[Union[str, List[str]]] = None,
    inplace: bool = False,
) -> DatasetTriplet:
    ds = ev.as_tf_dataset(select_columns=select_columns, inplace=inplace)

    return bulk_split(ds, splits, length=len(ev))
