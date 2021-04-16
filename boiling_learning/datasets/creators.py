import pprint
import warnings
from collections import defaultdict
from typing import Container, Optional, Sequence, Union

import dataclassy
import more_itertools as mit
from tensorflow.data.experimental import AUTOTUNE

import boiling_learning.preprocessing as bl_preprocessing
import boiling_learning.utils as bl_utils
from boiling_learning.datasets.datasets import (DatasetSplits,
                                                tf_concatenate,
                                                tf_train_val_test_split)
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.management.Manager import Manager
from boiling_learning.preprocessing.transformers import (Creator,
                                                         DictImageTransformer,
                                                         Transformer)
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import PathLike


@Creator.make('experiment_video_dataset_creator', expand_pack_on_call=True)
def experiment_video_dataset_creator(
        experiment_video: bl_preprocessing.ExperimentVideo,
        splits: DatasetSplits,
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
        splits: DatasetSplits,
        data_preprocessors: Sequence[Transformer],
        dataset_size: Optional[int] = None,
        num_shards: Optional[int] = None,
        verbose: int = 0,
        save: bool = True,
        load: bool = True,
        reload_after_save: bool = False
):
    experiment_video_dataset_params = bl_utils.Parameters(params=defaultdict(dict))
    experiment_video_dataset_params[['creator', {'desc', 'value'}, 'dataset_size']] = dataset_size
    experiment_video_dataset_params[['creator', {'desc', 'value'}, 'num_shards']] = num_shards
    experiment_video_dataset_params[['creator', 'desc', 'splits']] = dataclassy.as_dict(splits)
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
