from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Container, Optional, Sequence, Tuple

import dataclassy
import funcy
import more_itertools as mit

from boiling_learning.datasets import DatasetSplits, DatasetTriplet
from boiling_learning.io import load_json, save_json
from boiling_learning.io.io import (
    add_bool_flag,
    load_dataset,
    load_yogadl,
    loader_dataset_triplet,
    save_dataset,
    save_yogadl,
    saver_dataset_triplet,
)
from boiling_learning.management import Manager
from boiling_learning.preprocessing import ImageDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.utils import ensure_dir
from boiling_learning.utils.functional import P, Pack
from boiling_learning.utils.Parameters import Parameters


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
    take: Optional[int] = None,
    augment_train: bool = True,
    augment_test: bool = True,
    verbose: int = False,
    augmentors_to_force: Container[str] = frozenset({'random_cropper'}),
):
    def saver(ds: DatasetTriplet, path: Optional[Path] = None) -> None:
        success_path = ensure_dir(path) / 'success.json'

        if verbose:
            print('Saving to:', success_path)

        if success_path.is_file():
            if verbose:
                print('Is file:', success_path)
            json_data = load_json(success_path)
            if verbose:
                print('File contents:', json_data)
            success = json_data.get('success', False)
        else:
            success = False
            if verbose:
                print('File not found:', success_path)

        if not success:
            save_json({'success': False}, success_path)
            for split in ds:
                if ds is not None:
                    mit.consume(split.as_numpy_iterator())
            save_json({'success': True}, success_path)

    def loader(path: Optional[Path] = None) -> Tuple[bool, None]:
        return False, None

    if not augment_train:
        augmentors = [
            data_augmentor
            for data_augmentor in augmentors
            if data_augmentor.name in augmentors_to_force
        ]

    dataset_params = Parameters(params=defaultdict(dict))
    dataset_params[
        ['creator', {'desc', 'value'}, 'dataset_size']
    ] = dataset_size

    dataset_params[['creator', {'desc', 'value'}, 'num_shards']] = 1024

    dataset_params[['creator', 'desc', 'image_dataset']] = sorted(
        img_ds.keys()
    )
    dataset_params[['creator', 'value', 'image_dataset']] = img_ds

    print(f'Splits: {splits} ({type(splits)})')

    dataset_params[
        ['creator', {'desc', 'value'}, 'splits']
    ] = funcy.walk_values(str, dataclassy.asdict(splits))
    dataset_params[['creator', 'value', 'splits']] = splits

    dataset_params[['creator', 'desc', 'data_preprocessors']] = [
        data_preprocessor.describe() for data_preprocessor in preprocessors
    ]
    dataset_params[['creator', 'value', 'data_preprocessors']] = preprocessors

    dataset_params[['post_processor', 'desc', 'data_augmentors']] = [
        data_augmentor.describe() for data_augmentor in augmentors
    ]
    dataset_params[['post_processor', 'value', 'data_augmentors']] = augmentors

    dataset_params[
        ['post_processor', 'value', 'force_test_augmentors']
    ] = augmentors_to_force
    dataset_params[['post_processor', 'value', 'take']] = take

    dataset_params[
        ['creator', 'value', 'experiment_video_dataset_manager']
    ] = experiment_video_dataset_manager

    dataset_params[['creator', 'value', 'verbose']] = 2

    dataset_params[['creator', 'desc', 'save']] = {
        'name': 'bl.io.save_dataset',
        'params': P(),
    }
    dataset_params[['creator', 'value', 'save']] = saver_dataset_triplet(
        save_dataset
    )
    dataset_params[['creator', 'desc', 'load']] = {
        'name': 'bl.io.load_dataset',
        'params': P(),
    }
    dataset_params[['creator', 'value', 'load']] = loader_dataset_triplet(
        add_bool_flag(load_dataset)
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
        if None not in {shuffle_size, dataset_size}
        else shuffle_size
    )
    dataset_params[['post_processor', {'desc', 'value'}, 'batch_size']] = (
        min(batch_size, dataset_size)
        if None not in {batch_size, dataset_size}
        else batch_size
    )
    dataset_params[
        ['post_processor', {'desc', 'value'}, 'augment_test']
    ] = augment_test

    dataset_id = dataset_manager.provide_entry(
        creator_description=Pack(kwargs=dataset_params[['creator', 'desc']]),
        post_processor_description=Pack(
            kwargs=dataset_params[['post_processor', 'desc']]
        ),
        include=True,
        missing_ok=True,
    )

    # dataset_params[['creator', 'value', 'save']] = bl.io.saver_dataset_triplet(
    #     partial(bl.io.save_yogadl, dataset_id=dataset_id)
    # )
    # dataset_params[['creator', 'value', 'load']] = bl.io.loader_dataset_triplet(
    #     bl.io.add_bool_flag(
    #         partial(bl.io.load_yogadl, dataset_id=dataset_id, shuffle=load_shuffle, shuffle_seed=2020),
    #         (FileNotFoundError, AssertionError)
    #     )
    # )

    # workspace_path = dataset_manager.elem_workspace(dataset_id)
    # snapshot_path = workspace_path / 'snapshot'
    # dataset_params[['creator', 'value', 'snapshot_path']] = snapshot_path

    return dataset_id, dataset_manager.provide_elem(
        creator_description=Pack(kwargs=dataset_params[['creator', 'desc']]),
        creator_params=Pack(kwargs=dataset_params[['creator', 'value']]),
        post_processor_description=Pack(
            kwargs=dataset_params[['post_processor', 'desc']]
        ),
        post_processor_params=Pack(
            kwargs=dataset_params[['post_processor', 'value']]
        ),
        load=loader_dataset_triplet(
            add_bool_flag(
                partial(
                    load_yogadl,
                    dataset_id=dataset_id,
                    shuffle=shuffle,
                    shuffle_seed=2020,
                ),
                (FileNotFoundError, AssertionError),
            )
        ),
        save=saver_dataset_triplet(
            partial(save_yogadl, dataset_id=dataset_id)
        ),
    )


if __name__ == '__main__':
    raise RuntimeError(
        '*make_dataset* cannot be executed as a standalone script yet.'
    )
