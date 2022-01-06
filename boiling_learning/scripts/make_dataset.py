from collections import defaultdict
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Container, Optional, Sequence, Tuple, Union

import dataclassy
import funcy

from boiling_learning.datasets.datasets import DatasetSplits
from boiling_learning.datasets.sliceable import (
    SliceableDataset,
    load_sliceable_dataset,
    save_sliceable_dataset,
)
from boiling_learning.io.io import (
    DatasetTriplet,
    SaverFunction,
    add_bool_flag,
    load_dataset,
    load_yogadl,
    loader_dataset_triplet,
    save_dataset,
    save_yogadl,
    saver_dataset_triplet,
)
from boiling_learning.management.descriptors import describe
from boiling_learning.management.managers import Manager
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.utils.functional import Kwargs, P
from boiling_learning.utils.parameters import Parameters


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

    print(f'Splits: {splits} ({type(splits)})')

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

    if experiment_video_saver is None:
        experiment_video_saver = (
            saver_dataset_triplet(save_dataset)
            if as_tensors
            else saver_dataset_triplet(save_sliceable_dataset)
        )

    dataset_params[['creator', 'value', 'save']] = experiment_video_saver

    dataset_params[['creator', 'desc', 'load']] = {
        'name': 'bl.io.load_dataset',
        'params': P(),
    }

    dataset_params[['creator', 'value', 'load']] = (
        loader_dataset_triplet(add_bool_flag(load_dataset))
        if as_tensors
        else loader_dataset_triplet(add_bool_flag(load_sliceable_dataset))
    )

    # dataset_params[['creator', 'desc', 'save']] = {
    #     'name': 'bl.io.save_yogadl',
    #     'params': P()
    # }
    # dataset_params[['creator', 'desc', 'load']] = {
    #     'name': 'bl.io.load_yogadl',
    #     'params': P(shuffle=load_shuffle)
    # }
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
                partial(load_yogadl, dataset_id=dataset_id, shuffle=shuffle, shuffle_seed=2020),
                (FileNotFoundError, AssertionError),
            )
        )
        saver = saver_dataset_triplet(partial(save_yogadl, dataset_id=dataset_id))
    else:

        @loader_dataset_triplet
        @add_bool_flag
        def loader(path: Path) -> SliceableDataset[Any]:
            return load_sliceable_dataset(path).shuffle()

        saver = saver_dataset_triplet(save_sliceable_dataset)

    # dataset_params[['creator', 'value', 'save']] = bl.io.saver_dataset_triplet(
    #     partial(bl.io.save_yogadl, dataset_id=dataset_id)
    # )
    # dataset_params[['creator', 'value', 'load']] = bl.io.loader_dataset_triplet(
    #     bl.io.add_bool_flag(
    #         partial(
    #             bl.io.load_yogadl,
    #             dataset_id=dataset_id,
    #             shuffle=load_shuffle,
    #             shuffle_seed=2020
    #         ),
    #         (FileNotFoundError, AssertionError)
    #     )
    # )

    # workspace_path = dataset_manager.elem_workspace(dataset_id)
    # snapshot_path = workspace_path / 'snapshot'
    # dataset_params[['creator', 'value', 'snapshot_path']] = snapshot_path

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
