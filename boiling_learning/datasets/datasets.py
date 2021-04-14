import collections
import enum
import functools
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Optional, Union

import funcy
import more_itertools as mit
import tensorflow as tf
from sklearn.model_selection import train_test_split

import boiling_learning.utils.mathutils as mathutils
from boiling_learning.io.io import DatasetTriplet

_sentinel = object()


@dataclass(frozen=True)
class DatasetSplitter:
    train: Optional[Fraction] = None
    test: Optional[Fraction] = None
    val: Optional[Fraction] = Fraction(0)

    def __post_init__(self):
        splits = (self.train, self.val, self.test)
        n_nones = splits.count(None)
        if n_nones > 1:
            raise ValueError('at most one of *train*, *val* and *test* can be inferred (by passing None)')

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

        if not (
                0 < self.train < 1
                and 0 <= self.val < 1
                and 0 < self.test < 1
        ):
            raise ValueError('it is required that 0 < (*train*, *test*) < 1 and 0 <= *val* < 1')


class SplitSubset(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TRAIN_VAL = enum.auto()
    TEST = enum.auto()
    ALL = enum.auto()

    @classmethod
    def get_split(cls, s, default=_sentinel):
        if s in cls:
            return s
        else:
            return cls.from_string(s, default=default)

    @classmethod
    def from_string(cls, s, default=_sentinel):
        for k, v in cls.FROM_STR.items():
            if s in v:
                return k
        if default is _sentinel:
            raise ValueError(
                f'string {s} was not found in the conversion table.'
                f'Available values are {list(cls.FROM_STR.values())}.')
        else:
            return default

    def to_str(self):
        return self.name.lower()


SplitSubset.FROM_STR = {
    SplitSubset.TRAIN: {'train'},
    SplitSubset.VAL: {'val', 'validation'},
    SplitSubset.TRAIN_VAL: set(
        connector.join(['train', validation_key])
        for connector in ['_', '_and_']
        for validation_key in ['val', 'validation']
    ),
    SplitSubset.TEST: {'test'},
    SplitSubset.ALL: {'all'},
}


def train_val_test_split(
        dataset,
        n_samples,
        train_size=None,
        val_size=None,
        test_size=None,
        **options
):
    # TODO: this function only accepts one dataset. Allow more.
    # TODO: this function requires the argument n_samples. Remove this.
    # TODO: to keep consistency, allow elements from SplitSubset.FROM_STR

    if val_size is None or val_size == 0:
        train_set, test_set = train_test_split(
            dataset,
            train_size=train_size,
            test_size=test_size,
            **options
        )
        val_set = []
    else:
        if 0 < val_size < 1:
            val_size = int(val_size * n_samples)
        elif val_size < 0 or val_size > n_samples:
            raise ValueError(
                f'invalid val_size {val_size}.'
                'Expected a float in (0, 1),'
                f'or a float in [0, n_samples={n_samples}].')

        if train_size is None:
            train_set, test_set = train_test_split(
                dataset,
                test_size=test_size,
                **options
            )
            train_set, val_set = train_test_split(
                train_set,
                test_size=val_size,
                **options
            )
        else:
            train_set, test_set = train_test_split(
                dataset,
                train_size=train_size,
                **options
            )
            val_set, test_set = train_test_split(
                test_set,
                train_size=val_size,
                **options
            )

    return train_set, val_set, test_set


def tf_concatenate(datasets: Iterable[tf.data.Dataset]) -> tf.data.Dataset:
    datasets = collections.deque(datasets)

    if not datasets:
        raise ValueError('argument *datasets* must be a non-empty iterable.')

    ds = datasets.popleft()
    for dataset in datasets:
        ds = ds.concatenate(dataset)

    return ds


def tf_train_val_test_split(
        ds: tf.data.Dataset,
        splits: DatasetSplitter,
        shuffle: bool = False
) -> DatasetTriplet:
    """ #TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    This function is deterministic in the sense that it outputs the same splits given the same inputs.
    Consequently it is safe to be used early in the pipeline to avoid consuming test data.
    """
    def flatten_zip(*ds: tf.data.Dataset) -> tf.data.Dataset:
        if len(ds) == 1:
            return ds[0]
        else:
            return tf.data.Dataset.zip(ds)

    if splits.val == 0:
        split_train, split_test = mathutils.proportional_ints(splits.train, splits.test)
        window_shift = split_train + split_test
        if shuffle:
            ds = ds.shuffle(window_shift, reshuffle_each_iteration=False)
        ds_train = ds.window(split_train, window_shift).flat_map(flatten_zip)
        ds_test = ds.skip(split_train).window(split_test, window_shift).flat_map(flatten_zip)
        ds_val = None
    else:
        split_train, split_val, split_test = mathutils.proportional_ints(splits.train, splits.val, splits.test)
        window_shift = split_train + split_val + split_test
        if shuffle:
            ds = ds.shuffle(window_shift, reshuffle_each_iteration=False)
        ds_train = ds.window(split_train, window_shift).flat_map(flatten_zip)
        ds_val = ds.skip(split_train).window(split_val, window_shift).flat_map(flatten_zip)
        ds_test = ds.skip(split_train + split_val).window(split_test, window_shift).flat_map(flatten_zip)

    return ds_train, ds_val, ds_test


def tf_train_val_test_split_concat(
        datasets: Iterable[tf.data.Dataset],
        splits: DatasetSplitter
) -> DatasetTriplet:
    """ #TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    The paramenter *splits* is must contain only integers, with the possible exception of *val*, which can be *None*.

    This function is deterministic in the sense that it outputs the same splits given the same inputs.
    Consequently it is safe to be used early in the pipeline to avoid consuming test data.
    """
    datasets = collections.deque(datasets)

    if not datasets:
        raise ValueError('argument *datasets* must be a non-empty iterable.')

    ds = datasets.popleft()
    ds_train, ds_val, ds_test = tf_train_val_test_split(ds, splits)

    if ds_val is None:
        for ds in datasets:
            ds_train_, _, ds_test_ = tf_train_val_test_split(ds, splits)
            ds_train = ds_train.concatenate(ds_train_)
            ds_test = ds_test.concatenate(ds_test_)
    else:
        for ds in datasets:
            ds_train_, ds_val_, ds_test_ = tf_train_val_test_split(ds, splits)
            ds_train = ds_train.concatenate(ds_train_)
            ds_val = ds_val.concatenate(ds_val_)
            ds_test = ds_test.concatenate(ds_test_)

    return ds_train, ds_val, ds_test


def take(
        ds: tf.data.Dataset,
        count: Optional[Union[int, Fraction]],
        unbatch_dim: Optional[int] = None,
        unbatch_key: Optional[int] = None
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
            functools.partial(take, count=count, unbatch_dim=None, unbatch_key=None),
            dim=unbatch_dim,
            key=unbatch_key
        )

    if isinstance(count, int):
        return ds.take(count)
    elif isinstance(count, Fraction):
        return tf_train_val_test_split(
            ds,
            splits=DatasetSplitter(train=count)
        )[0]
    else:
        raise TypeError(
            f'*count* must be either *int* or *Fraction*, got {type(count)}.'
        )


def calculate_batch_size(
        dataset: tf.data.Dataset,
        dim: int = 0,
        key: Optional[int] = None
) -> int:
    elem = mit.first(dataset)
    if key is not None:
        elem = elem[key]
    return elem.shape[dim]


def calculate_dataset_size(
        dataset: tf.data.Dataset,
        dim: int = 0,
        is_batched: bool = False
) -> int:
    if is_batched:
        batch_size_calculator = functools.partial(calculate_batch_size, dim=dim)
        return sum(
            map(
                batch_size_calculator,
                dataset
            )
        )
    else:
        try:
            return len(dataset)
        except TypeError:
            return mit.ilen(dataset)


def apply_unbatched(
        dataset: tf.data.Dataset,
        apply: Callable[[tf.data.Dataset], tf.data.Dataset],
        dim: int = 0,
        key: Optional[int] = None
) -> tf.data.Dataset:
    batch_size = calculate_batch_size(dataset, dim=dim, key=key)
    return dataset.unbatch().apply(apply).batch(batch_size)


def map_unbatched(
        dataset: tf.data.Dataset,
        map_fn: Callable,
        dim: int = 0,
        key: Optional[int] = None
) -> tf.data.Dataset:
    return apply_unbatched(
        dataset,
        lambda ds: ds.map(map_fn),
        dim=dim,
        key=key
    )


def filter_unbatched(
        dataset: tf.data.Dataset,
        pred_fn: Callable[..., bool],
        dim: int = 0,
        key: Optional[int] = None
) -> tf.data.Dataset:
    return apply_unbatched(
        dataset,
        lambda ds: ds.filter(pred_fn),
        dim=dim,
        key=key
    )
