import collections
from dataclasses import dataclass
import enum
from typing import (
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    Union
)

import parse
from sklearn.model_selection import train_test_split
import tensorflow as tf

import boiling_learning.utils.utils as bl_utils
from boiling_learning.utils.utils import PathType
import boiling_learning.io.io as bl_io

_sentinel = object()
T = TypeVar('T')


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


@dataclass
class DatasetSplitter:
    train: Optional[Union[int, float]]
    test: Optional[Union[int, float]]
    val: Optional[Union[int, float]] = None
    n_samples: Optional[int] = None

    # def __post_init__(self):
    #     if self.n_samples is not None:
    #         if self.train is not None and 0 < self.train < 1:
    #             self.train *= self.n_samples
    #         if self.val is not None and 0 < self.val < 1:
    #             self.val *= self.n_samples
    #         if self.test is not None and 0 < self.test < 1:
    #             self.test *= self.n_samples


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
        splits: DatasetSplitter
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset]:
    """ #TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    The paramenter *splits* is expected to contain only integers, with the possible exception of *val*, which can be *None*.

    This function is deterministic in the sense that it outputs the same splits given the same inputs.
    Consequently it is safe to be used early in the pipeline to avoid consuming test data.
    """
    if not isinstance(splits.train, int) or not isinstance(splits.test, int):
        raise ValueError(
            '*splits.train* and *splits.test* must be *int*s.'
            ' *splits.val* is allowed to be *None*, and in this case no validation set is produced (*None* is returned)')

    def flatten_zip(*ds):
        if len(ds) == 1:
            return ds[0]
        else:
            return tf.data.Dataset.zip(ds)

    if isinstance(splits.val, int):
        window_shift = splits.train + splits.val + splits.test
        ds_train = ds.window(splits.train, window_shift).flat_map(flatten_zip)
        ds_val = ds.skip(splits.train).window(splits.val, window_shift).flat_map(flatten_zip)
        ds_test = ds.skip(splits.train + splits.val).window(splits.test, window_shift).flat_map(flatten_zip)
    else:
        window_shift = splits.train + splits.test
        ds_train = ds.window(splits.train, window_shift).flat_map(flatten_zip)
        ds_test = ds.skip(splits.train).window(splits.test, window_shift).flat_map(flatten_zip)
        ds_val = None

    return ds_train, ds_val, ds_test


def tf_train_val_test_split_concat(
        datasets: Iterable[tf.data.Dataset],
        splits: DatasetSplitter
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset]:
    """ #TODO(docstring): describe here

    # Source: <https://stackoverflow.com/a/60503037/5811400>

    The paramenter *splits* is expected to contain only integers, with the possible exception of *val*, which can be *None*.

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


def restore(
    restore: bool = False,
    path: Optional[PathType] = None,
    load_method: bl_io.LoaderFunction[T] = None,
    epoch_str: str = 'epoch'
) -> Tuple[int, Optional[T]]:
    last_epoch = -1
    model = None
    if restore:
        path = bl_utils.ensure_resolved(path)
        glob_pattern = path.name.replace(f'{{{epoch_str}}}', '*')
        parser = parse.compile(path.name).parse

        paths = path.parent.glob(glob_pattern)
        parsed = (parser(path_item.name) for path_item in paths)
        parsed = filter(lambda p: p is not None and epoch_str in p, parsed)
        epochs = bl_utils.append(
            (int(p[epoch_str]) for p in parsed),
            last_epoch
        )
        last_epoch = max(epochs)

        if last_epoch != -1:
            path_str = str(path).format(epoch=last_epoch)
            model = load_method(path_str)

    return last_epoch, model
