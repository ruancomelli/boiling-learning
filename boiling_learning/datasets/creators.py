import pprint
import warnings
from collections import defaultdict
from fractions import Fraction
from typing import Container, Iterable, Optional, Sequence, Union

import dataclassy
import more_itertools as mit
from tensorflow.data import AUTOTUNE

from boiling_learning.datasets.datasets import (
    DatasetSplits,
    apply_transformers,
    concatenate,
    experiment_video_to_dataset_triplet,
    take,
)
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.management.Manager import Manager
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.ImageDataset import ImageDataset
from boiling_learning.preprocessing.preprocessing import snapshotter
from boiling_learning.preprocessing.transformers import (
    DictFeatureTransformer,
    Transformer,
    creator,
    transformer,
)
from boiling_learning.utils.functional import Kwargs, Pack
from boiling_learning.utils.Parameters import Parameters
from boiling_learning.utils.utils import PathLike, elapsed_timer, ensure_dir


@creator(expand_pack_on_call=True)
def experiment_video_dataset_creator(
    experiment_video: ExperimentVideo,
    splits: DatasetSplits,
    data_preprocessors: Iterable[Transformer],
    dataset_size: Optional[Union[int, Fraction]] = None,
    snapshot_path: Optional[PathLike] = None,
    num_shards: Optional[int] = None,
) -> DatasetTriplet:
    ds_triplet = experiment_video_to_dataset_triplet(
        experiment_video, splits=splits, dataset_size=dataset_size
    )

    data_preprocessors = tuple(data_preprocessors)
    ds_triplet = apply_transformers(ds_triplet, data_preprocessors)

    ds_train, ds_val, ds_test = ds_triplet

    if snapshot_path is not None:
        snapshot_path = ensure_dir(snapshot_path)

        if isinstance(dataset_size, int):
            num_shards = min(dataset_size, num_shards)
        elif isinstance(dataset_size, Fraction):
            num_shards = min(int(dataset_size * len(experiment_video)), num_shards)

        ds_train = ds_train.apply(
            snapshotter(
                snapshot_path / 'train',
                num_shards=num_shards,
                shuffle_size=num_shards,
            )
        )
        if ds_val is not None:
            ds_val = ds_val.apply(
                snapshotter(
                    snapshot_path / 'val',
                    num_shards=num_shards,
                    shuffle_size=num_shards,
                )
            )
        ds_test = ds_test.apply(
            snapshotter(
                snapshot_path / 'test',
                num_shards=num_shards,
                shuffle_size=num_shards,
            )
        )

    return ds_train, ds_val, ds_test


@creator(expand_pack_on_call=True)
def dataset_creator(
    experiment_video_dataset_manager: Manager,
    image_dataset: ImageDataset,
    splits: DatasetSplits,
    data_preprocessors: Sequence[Transformer],
    dataset_size: Optional[Union[int, Fraction]] = None,
    num_shards: Optional[int] = None,
    verbose: int = 0,
    save: bool = True,
    load: bool = True,
    reload_after_save: bool = False,
    as_tensors: bool = False,
):
    experiment_video_dataset_params = Parameters(params=defaultdict(dict))
    experiment_video_dataset_params[['creator', {'desc', 'value'}, 'dataset_size']] = dataset_size
    experiment_video_dataset_params[['creator', {'desc', 'value'}, 'num_shards']] = num_shards
    experiment_video_dataset_params[['creator', 'desc', 'splits']] = dataclassy.as_dict(splits)
    experiment_video_dataset_params[['creator', 'value', 'splits']] = splits

    ds_dict = {}
    for name, ev in image_dataset.items():
        _data_preprocessors = [
            data_preprocessor[name]
            if isinstance(data_preprocessor, DictFeatureTransformer)
            else data_preprocessor
            for data_preprocessor in data_preprocessors
        ]
        experiment_video_dataset_params[['creator', 'desc', 'experiment_video']] = ev.name
        experiment_video_dataset_params[['creator', 'value', 'experiment_video']] = ev
        experiment_video_dataset_params[['creator', 'desc', 'data_preprocessors']] = [
            data_preprocessor.describe() for data_preprocessor in _data_preprocessors
        ]
        experiment_video_dataset_params[
            ['creator', 'value', 'data_preprocessors']
        ] = _data_preprocessors
        dataset_id = experiment_video_dataset_manager.provide_entry(
            creator_description=Kwargs(experiment_video_dataset_params[['creator', 'desc']]),
            post_processor_description=Pack(),
            include=True,
            missing_ok=True,
        )
        workspace_path = experiment_video_dataset_manager.elem_workspace(dataset_id)
        experiment_video_dataset_params[['creator', 'value', 'snapshot_path']] = (
            workspace_path / 'snapshot'
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ResourceWarning)

            with elapsed_timer() as timer:
                ds_dict[name] = experiment_video_dataset_manager.provide_elem(
                    creator_description=Kwargs(
                        experiment_video_dataset_params[['creator', 'desc']]
                    ),
                    creator_params=Kwargs(experiment_video_dataset_params[['creator', 'value']]),
                    save=save,
                    load=load,
                    reload_after_save=reload_after_save,
                )
            print(name, 'took', timer.duration)

    if verbose:
        print('--- ds_dict ---')
        pprint.pprint(ds_dict)

    datasets_train, datasets_val, datasets_test = map(tuple, mit.unzip(ds_dict.values()))

    ds_train = concatenate(datasets_train)
    ds_val = concatenate(datasets_val) if None not in datasets_val else None
    ds_test = concatenate(datasets_test)

    if isinstance(dataset_size, int):
        ds_train = take(ds_train, dataset_size)
        ds_val = take(ds_val, dataset_size)
        ds_test = take(ds_test, dataset_size)

    return (ds_train, ds_val, ds_test)


@transformer()
def dataset_post_processor(
    ds: DatasetTriplet,
    data_augmentors: Iterable[Transformer],
    cache: Union[bool, PathLike] = False,
    batch_size: Optional[int] = None,
    prefetch: bool = True,
    shuffle_size: Optional[int] = None,
    augment_test: bool = False,
    force_test_augmentors: Container[str] = frozenset(),
    take: Optional[int] = None,
    verbose: bool = False,
):
    data_augmentors = tuple(data_augmentors)

    if verbose:
        print('>>> Datasets:', ds)
        print('>>> Data augmentors:', data_augmentors)

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
        cache = ensure_dir(cache)
        ds_train = ds_train.cache(str(cache / 'train'))
        if ds_val is not None:
            ds_val = ds_val.cache(str(cache / 'val'))
        ds_test = ds_test.cache(str(cache / 'test'))

    ds_train = apply_transformers(ds_train, data_augmentors)

    test_augmentors = (
        data_augmentors
        if augment_test
        else tuple(
            data_augmentor
            for data_augmentor in data_augmentors
            if data_augmentor.name in force_test_augmentors
        )
    )
    ds_val = apply_transformers(ds_val, test_augmentors)
    ds_test = apply_transformers(ds_test, test_augmentors)

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

    return ds_train, ds_val, ds_test
