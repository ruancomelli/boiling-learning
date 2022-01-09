from collections import defaultdict
from fractions import Fraction
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Container, Optional, Sequence, Tuple, Union

import dataclassy
import funcy

from boiling_learning.datasets.datasets import DatasetSplits
from boiling_learning.datasets.sliceable import (
    SliceableDataset,
    load_supervised_sliceable_dataset,
    save_supervised_sliceable_dataset,
)
from boiling_learning.io import json
from boiling_learning.io.io import (
    DatasetTriplet,
    SaverFunction,
    add_bool_flag,
    load_dataset,
    load_image,
    load_yogadl,
    loader_dataset_triplet,
    save_dataset,
    save_image,
    save_yogadl,
    saver_dataset_triplet,
)
from boiling_learning.management.managers import Manager
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.descriptors import describe
from boiling_learning.utils.functional import Kwargs, P
from boiling_learning.utils.parameters import Parameters
from boiling_learning.utils.utils import resolve


def main(
    experiment_video_dataset_manager: Manager,
    dataset_manager: Manager,
    img_ds: ImageDataset,
    splits: DatasetSplits,
    preprocessors: Sequence[Transformer],
    augmentors: Sequence[Transformer],
    dataset_size: Optional[int] = None,
    shuffle: bool = True,
    shuffle_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    take: Optional[Union[int, Fraction]] = None,
    augment_train: bool = True,
    augment_test: bool = True,
    verbose: int = False,
    augmentors_to_force: Container[str] = frozenset({'random_cropper'}),
    experiment_video_saver: Optional[SaverFunction[DatasetTriplet]] = None,
    as_tensors: bool = False,
) -> Tuple[str, DatasetTriplet]:
    if not augment_train:
        augmentors = [
            data_augmentor
            for data_augmentor in augmentors
            if data_augmentor.name in augmentors_to_force
        ]

    dataset_params = Parameters(params=defaultdict(dict))
    dataset_params[['creator', {'desc', 'value'}, 'dataset_size']] = dataset_size

    if as_tensors:
        dataset_params[['creator', {'desc', 'value'}, 'num_shards']] = 1024
    else:
        dataset_params[['creator', {'desc', 'value'}, 'as_tensors']] = as_tensors

    dataset_params[['creator', 'desc', 'image_dataset']] = sorted(img_ds.keys())
    dataset_params[['creator', 'value', 'image_dataset']] = img_ds

    dataset_params[['creator', {'desc', 'value'}, 'splits']] = funcy.walk_values(
        str, dataclassy.asdict(splits)
    )
    dataset_params[['creator', 'value', 'splits']] = splits

    dataset_params[['creator', 'desc', 'data_preprocessors']] = describe(preprocessors)
    dataset_params[['creator', 'value', 'data_preprocessors']] = preprocessors

    dataset_params[['post_processor', 'desc', 'data_augmentors']] = describe(augmentors)
    dataset_params[['post_processor', 'value', 'data_augmentors']] = augmentors

    dataset_params[['post_processor', 'value', 'force_test_augmentors']] = augmentors_to_force
    dataset_params[['post_processor', 'value', 'take']] = take

    dataset_params[
        ['creator', 'value', 'experiment_video_dataset_manager']
    ] = experiment_video_dataset_manager

    dataset_params[['creator', 'value', 'verbose']] = 2

    dataset_params[['creator', 'desc', 'save']] = {
        'name': 'bl.io.save_dataset',
        'params': P(),
    }

    def _feature_saver(image: VideoFrame, path: PathLike) -> None:
        save_image(image, resolve(path).with_suffix('.png'))

    def _feature_loader(path: PathLike) -> None:
        return load_image(resolve(path).with_suffix('.png'))

    sliceable_dataset_saver = partial(
        save_supervised_sliceable_dataset, feature_saver=_feature_saver, target_saver=json.dump
    )

    sliceable_dataset_loader = partial(
        load_supervised_sliceable_dataset, feature_loader=_feature_loader, target_loader=json.load
    )

    if experiment_video_saver is None:
        experiment_video_saver = (
            saver_dataset_triplet(save_dataset)
            if as_tensors
            else saver_dataset_triplet(sliceable_dataset_saver)
        )

    dataset_params[['creator', 'value', 'save']] = experiment_video_saver

    dataset_params[['creator', 'desc', 'load']] = {
        'name': 'bl.io.load_dataset',
        'params': P(),
    }

    dataset_params[['creator', 'value', 'load']] = (
        loader_dataset_triplet(add_bool_flag(load_dataset))
        if as_tensors
        else loader_dataset_triplet(add_bool_flag(sliceable_dataset_loader))
    )

    dataset_params[['creator', {'desc', 'value'}, 'reload_after_save']] = True

    dataset_params[['post_processor', {'desc', 'value'}, 'prefetch']] = True
    dataset_params[['post_processor', {'desc', 'value'}, 'shuffle_size']] = (
        min(shuffle_size, dataset_size)
        if None not in {shuffle_size, dataset_size} and isinstance(dataset_size, int)
        else shuffle_size
    )
    dataset_params[['post_processor', {'desc', 'value'}, 'batch_size']] = (
        min(batch_size, dataset_size)
        if None not in {batch_size, dataset_size} and isinstance(dataset_size, int)
        else batch_size
    )
    dataset_params[['post_processor', {'desc', 'value'}, 'augment_test']] = augment_test

    dataset_id = dataset_manager.provide_entry(
        creator_description=Kwargs(dataset_params[['creator', 'desc']]),
        post_processor_description=Kwargs(dataset_params[['post_processor', 'desc']]),
        include=True,
        missing_ok=True,
    )

    if as_tensors:
        loader = loader_dataset_triplet(
            add_bool_flag(
                partial(load_yogadl, dataset_id=dataset_id, shuffle=shuffle, shuffle_seed=1997),
                (FileNotFoundError, AssertionError),
            )
        )
        saver = saver_dataset_triplet(partial(save_yogadl, dataset_id=dataset_id))
    else:

        @loader_dataset_triplet
        @add_bool_flag
        def loader(path: Path) -> SliceableDataset[Any]:
            return sliceable_dataset_loader(path).shuffle()

        saver = saver_dataset_triplet(sliceable_dataset_saver)

    return dataset_id, dataset_manager.provide_elem(
        creator_description=Kwargs(dataset_params[['creator', 'desc']]),
        creator_params=Kwargs(dataset_params[['creator', 'value']]),
        post_processor_description=Kwargs(dataset_params[['post_processor', 'desc']]),
        post_processor_params=Kwargs(dataset_params[['post_processor', 'value']]),
        load=loader,
        save=saver,
    )


if __name__ == '__main__':
    raise RuntimeError('*make_dataset* cannot be executed as a standalone script yet.')
