import collections
import dataclasses
import enum
import functools
import pprint
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Container, Iterable, Optional, Sequence, Union

import funcy
import more_itertools as mit
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.data.experimental import AUTOTUNE

import boiling_learning.preprocessing as bl_preprocessing
import boiling_learning.utils as bl_utils
import boiling_learning.utils.mathutils as mathutils
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.management.Manager import Manager
from boiling_learning.preprocessing.transformers import (
    Creator,
    DictImageTransformer,
    Transformer
)
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import PathLike

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


@Creator.make('experiment_video_dataset_creator', expand_pack_on_call=True)
def experiment_video_dataset_creator(
        experiment_video: bl_preprocessing.ExperimentVideo,
        splits: DatasetSplitter,
        data_preprocessors: Sequence[Transformer],
        dataset_size: Optional[int] = None,
        snapshot_path: Optional[PathLike] = None,
        num_shards: Optional[int] = None
):
    ds = experiment_video.as_tf_dataset()

    for t in data_preprocessors:
        ds = ds.map(
            t.as_tf_py_function(pack_tuple=True),
            num_parallel_calls=AUTOTUNE
        )

    ds_train, ds_val, ds_test = tf_train_val_test_split(ds, splits)

    if dataset_size is not None:
        ds_train = ds_train.take(dataset_size)
        if ds_val is not None:
            ds_val = ds_val.take(dataset_size)
        ds_test = ds_test.take(dataset_size)

    if snapshot_path is not None:
        snapshot_path = bl_utils.ensure_dir(snapshot_path)

        if dataset_size is not None:
            num_shards = min([dataset_size, num_shards])

        ds_train = ds_train.apply(
            bl_preprocessing.snapshotter(
                snapshot_path / 'train',
                num_shards=num_shards,
                shuffle_size=num_shards
            )
        )
        if ds_val is not None:
            ds_val = ds_val.apply(
                bl_preprocessing.snapshotter(
                    snapshot_path / 'val',
                    num_shards=num_shards,
                    shuffle_size=num_shards
                )
            )
        ds_test = ds_test.apply(
            bl_preprocessing.snapshotter(
                snapshot_path / 'test',
                num_shards=num_shards,
                shuffle_size=num_shards
            )
        )

    return (ds_train, ds_val, ds_test)


@Creator.make('dataset_creator', expand_pack_on_call=True)
def dataset_creator(
        experiment_video_dataset_manager: Manager,
        image_dataset: bl_preprocessing.ImageDataset,
        splits: DatasetSplitter,
        data_preprocessors: Sequence[Transformer],
        dataset_size: Optional[int] = None,
        snapshot_path: Optional[PathLike] = None,
        num_shards: Optional[int] = None,
        verbose: int = 0,
        save: bool = True,
        load: bool = True,
        reload_after_save: bool = False
):
    experiment_video_dataset_params = bl_utils.Parameters(params=collections.defaultdict(dict))
    experiment_video_dataset_params[['creator', {'desc', 'value'}, 'dataset_size']] = dataset_size
    experiment_video_dataset_params[['creator', {'desc', 'value'}, 'num_shards']] = num_shards
    experiment_video_dataset_params[['creator', 'desc', 'splits']] = dataclasses.asdict(splits)
    experiment_video_dataset_params[['creator', 'value', 'splits']] = splits

    ds_dict = {}
    for name, ev in image_dataset.items():
        data_preprocessors = [
            data_preprocessor[name]
            if isinstance(data_preprocessor, DictImageTransformer)
            else data_preprocessor
            for data_preprocessor in data_preprocessors
        ]
        experiment_video_dataset_params[['creator', 'desc', 'experiment_video']] = ev.name
        experiment_video_dataset_params[['creator', 'value', 'experiment_video']] = ev
        experiment_video_dataset_params[['creator', 'desc', 'data_preprocessors']] = [
            data_preprocessor.describe() for data_preprocessor in data_preprocessors
        ]
        experiment_video_dataset_params[['creator', 'value', 'data_preprocessors']] = data_preprocessors
        dataset_id = experiment_video_dataset_manager.provide_entry(
            creator_description=Pack(kwargs=experiment_video_dataset_params[['creator', 'desc']]),
            post_processor_description=Pack(),
            include=True,
            missing_ok=True
        )
        workspace_path = experiment_video_dataset_manager.elem_workspace(dataset_id)
        experiment_video_dataset_params[['creator', 'value', 'snapshot_path']] = workspace_path / 'snapshot'

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ResourceWarning)

            with bl_utils.elapsed_timer() as timer:
                ds_dict[name] = experiment_video_dataset_manager.provide_elem(
                    creator_description=Pack(kwargs=experiment_video_dataset_params[['creator', 'desc']]),
                    creator_params=Pack(kwargs=experiment_video_dataset_params[['creator', 'value']]),
                    save=save,
                    load=load,
                    reload_after_save=reload_after_save
                )
            print(name, 'took', timer.duration)

    if verbose:
        print('--- ds_dict ---')
        pprint.pprint(ds_dict)

    datasets_train, datasets_val, datasets_test = map(tuple, mit.unzip(ds_dict.values()))

    ds_train = tf_concatenate(datasets_train)
    if None in datasets_val:
        ds_val = None
    else:
        ds_val = tf_concatenate(datasets_val)
    ds_test = tf_concatenate(datasets_test)

    if dataset_size is not None:
        ds_train = ds_train.take(dataset_size)
        if ds_val is not None:
            ds_val = ds_val.take(dataset_size)
        ds_test = ds_test.take(dataset_size)

    return (ds_train, ds_val, ds_test)


@Transformer.make('dataset_post_processor')
def dataset_post_processor(
        ds: DatasetTriplet,
        data_augmentors: Sequence[Transformer],
        cache: Union[bool, PathLike] = False,
        batch_size: Optional[int] = None,
        prefetch: bool = True,
        shuffle_size: Optional[int] = None,
        augment_test: bool = False,
        force_test_augmentors: Container[str] = frozenset(),
        take: Optional[int] = None,
        verbose: bool = False
):
    if verbose:
        print('>>>> Datasets:', ds)
        print('>>>> Data augmentors:', data_augmentors)

    ds_train, ds_val, ds_test = ds
    if take is not None:
        ds_train = ds_train.take(take)
        if ds_val is not None:
            ds_val = ds_val.take(take)
        ds_test = ds_test.take(take)

    if isinstance(cache, bool):
        if cache:
            ds_train = ds_train.cache()
            if ds_val is not None:
                ds_val = ds_val.cache()
            ds_test = ds_test.cache()
    else:
        cache = bl_utils.ensure_dir(cache)
        ds_train = ds_train.cache(str(cache / 'train'))
        if ds_val is not None:
            ds_val = ds_val.cache(str(cache / 'val'))
        ds_test = ds_test.cache(str(cache / 'test'))

    for data_augmentor in data_augmentors:
        ds_train = ds_train.map(
            data_augmentor.as_tf_py_function(pack_tuple=True),
            num_parallel_calls=AUTOTUNE
        )

        if augment_test or data_augmentor.name in force_test_augmentors:
            if ds_val is not None:
                ds_val = ds_val.map(
                    data_augmentor.as_tf_py_function(pack_tuple=True),
                    num_parallel_calls=AUTOTUNE
                )
            ds_test = ds_train.map(
                data_augmentor.as_tf_py_function(pack_tuple=True),
                num_parallel_calls=AUTOTUNE
            )

    if shuffle_size is not None:
        ds_train = ds_train.shuffle(shuffle_size)
        if ds_val is not None:
            ds_val = ds_val.shuffle(shuffle_size)
        ds_test = ds_test.shuffle(shuffle_size)

    if batch_size is not None:
        ds_train = ds_train.batch(batch_size)
        if ds_val is not None:
            ds_val = ds_val.batch(batch_size)
        ds_test = ds_test.batch(batch_size)

    if prefetch:
        ds_train = ds_train.prefetch(AUTOTUNE)
        if ds_val is not None:
            ds_val = ds_val.prefetch(AUTOTUNE)
        ds_test = ds_test.prefetch(AUTOTUNE)

    return (ds_train, ds_val, ds_test)


# class SplitterPackTransformerEncoder(PackTransformerEncoder):
#     def default(self, obj):
#         if isinstance(obj, DatasetSplitter):
#             return funcy.walk_values(str, dataclasses.asdict(obj))
#         if isinstance(obj, Fraction):
#             return str(obj)
#         # Let the base class default method raise the TypeError
#         return PackTransformerEncoder.default(self, obj)


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
