import os
from contextlib import suppress
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Dict, ItemsView, KeysView, NamedTuple, ValuesView

import funcy
import json_tricks
import ray
import tensorflow as tf
import tensorflow_addons as tfa
from dotenv import dotenv_values

import boiling_learning as bl
from boiling_learning.datasets.creators import (
    dataset_creator,
    dataset_post_processor,
    experiment_video_dataset_creator,
)
from boiling_learning.datasets.sliceable import (
    SliceableDataset,
    sliceable_dataset_to_tensorflow_dataset,
)
from boiling_learning.management.managers import Manager
from boiling_learning.model.definitions import HoboldNet3
from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.scripts import (
    load_cases,
    load_dataset_tree,
    make_boiling_processors,
    make_condensation_processors,
    make_dataset,
    make_model,
    set_boiling_cases_data,
    set_condensation_datasets_data,
)
from boiling_learning.scripts.utils.initialization import check_all_paths_exist, initialize_gpus
from boiling_learning.utils.lazy import Lazy, LazyCallable
from boiling_learning.utils.typeutils import Many, typename
from boiling_learning.utils.utils import print_header, resolve

ray.init()


print_header('Initializing script')


class Options(NamedTuple):
    login_user: bool = False
    convert_videos: bool = True
    save_frames_to_tensor: bool = False
    extract_audios: bool = False
    extract_frames: bool = False
    pre_load_videos: bool = False
    interact_processed_frames: bool = False
    analyze_downsampling: bool = False
    analyze_consecutive_frames: bool = False
    analyze_learning_curve: bool = True
    analyze_cross_evaluation: bool = True

    def keys(self) -> KeysView[str]:
        return self._asdict().keys()

    def values(self) -> ValuesView[Any]:
        return self._asdict().values()

    def items(self) -> ItemsView[str, Any]:
        return self._asdict().items()


print_header('Options', level=1)
OPTIONS = Options()
for option, value in OPTIONS.items():
    print(f'{option}: {value}')

config: Dict[str, Any] = {}
with suppress(FileNotFoundError):
    config.update(dotenv_values('.env'))

boiling_learning_path: Path = resolve(os.environ['BOILING_DATA_PATH'])
boiling_experiments_path: Path = boiling_learning_path / 'experiments'
boiling_cases_path: Path = boiling_learning_path / 'cases'
condensation_learning_path: Path = resolve(os.environ['CONDENSATION_DATA_PATH'])
condensation_cases_path: Path = condensation_learning_path / 'data'
analyses_path: Path = boiling_learning_path / 'analyses'

print_header('Important paths', level=1)
check_all_paths_exist(
    (
        ('Boiling learning', boiling_learning_path),
        ('Boiling cases', boiling_cases_path),
        ('Boiling experiments', boiling_experiments_path),
        ('Contensation learning', condensation_learning_path),
        ('Contensation cases', condensation_cases_path),
        ('Analyses', analyses_path),
    ),
    verbose=True,
)

print_header('Checking CPUs and GPUs', level=1)
strategy: tf.distribute.Strategy = initialize_gpus()
strategy_name: str = typename(strategy)
print('Using distribute strategy:', strategy_name)

boiling_cases_names: Many[str] = tuple(f'case {idx+1}' for idx in range(5))
boiling_cases_names_timed: Many[str] = tuple(funcy.without(boiling_cases_names, 'case 1'))

print_header('Preparing datasets')
print_header('Loading cases', level=1)
print('Loading boiling cases from', boiling_cases_path)
boiling_cases: Lazy[Many[Case]] = LazyCallable(load_cases.main)(
    (boiling_cases_path / case_name for case_name in boiling_cases_names),
    video_suffix='.MP4',
    options=load_cases.Options(
        convert_videos=OPTIONS.convert_videos,
        extract_audios=OPTIONS.extract_audios,
        pre_load_videos=OPTIONS.pre_load_videos,
        extract_frames=OPTIONS.extract_frames,
    ),
    verbose=False,
)
boiling_cases_timed: Lazy[Many[Case]] = Lazy(
    lambda: tuple(case for case in boiling_cases() if case.name in boiling_cases_names_timed)
)

boiling_experiments_map: Dict[str, Path] = {
    'case 1': boiling_experiments_path / 'Experiment 2020-08-03 16-19' / 'data.csv',
    'case 2': boiling_experiments_path / 'Experiment 2020-08-05 14-15' / 'data.csv',
    'case 3': boiling_experiments_path / 'Experiment 2020-08-05 17-02' / 'data.csv',
    'case 4': boiling_experiments_path / 'Experiment 2020-08-28 15-28' / 'data.csv',
    'case 5': boiling_experiments_path / 'Experiment 2020-09-10 13-53' / 'data.csv',
}

print('Loading condensation cases from', condensation_cases_path)
condensation_datasets = LazyCallable(load_dataset_tree.main)(
    condensation_cases_path,
    load_dataset_tree.Options(
        convert_videos=OPTIONS.convert_videos,
        extract_audios=OPTIONS.extract_audios,
        pre_load_videos=OPTIONS.pre_load_videos,
        extract_frames=OPTIONS.extract_frames,
    ),
)

print_header('Setting up video data', level=1)
print('Setting boiling data from experiments path:', boiling_experiments_path)
set_boiling_cases_data.main(
    boiling_cases_timed(),
    case_experiment_map=boiling_experiments_map,
    verbose=True,
)

condensation_data_path = condensation_cases_path / 'data_spec.yaml'
print('Setting condensation data from data path:', condensation_data_path)
condensation_datasets_dict = set_condensation_datasets_data.main(
    condensation_datasets(),
    condensation_data_path,
    verbose=2,
    fps_cache_path=Path('.cache', 'fps'),
)
condensation_all_cases = ImageDataset.make_union(*condensation_datasets_dict.values())

boiling_preprocessors, boiling_augmentors = make_boiling_processors.main(
    direct_visualization=True,
    downscale_factor=5,
    direct_height=180,
    indirect_height=108,
    indirect_height_ratio=0.4,
    width=128,
)

condensation_preprocessors, condensation_augmentors = make_condensation_processors.main(
    downscale_factor=5, height=8 * 12, width=8 * 12
)


def table_saver(obj: Dict[str, Any], path: Path) -> None:
    path = resolve(path, parents=True)
    json_tricks.dump(obj, str(path))


def table_loader(path: Path) -> None:
    path = resolve(path, parents=True)
    return json_tricks.load(str(path))


description_comparer = partial(
    bl.utils.json_equivalent, dumps=json_tricks.dumps, loads=json_tricks.loads
)

save_map = {
    'model': tf.keras.models.save_model,
    'history': lambda obj, path: bl.io.save_pkl(getattr(obj, 'history', obj), path),
}
load_map = {
    'model': lambda path: bl.io.load_keras_model(
        path,
        strategy,
        custom_objects={
            'AdditionalValidationSets': bl.model.callbacks.AdditionalValidationSets,
            'RSquare': tfa.metrics.RSquare,
        },
    ),
    'history': bl.io.load_pkl,
}

experiment_video_dataset_manager = Manager(
    path=analyses_path / 'datasets' / 'boiling_experiment_video_datasets',
    id_fmt='experiment video dataset {index}',
    index_key='index',
    creator=experiment_video_dataset_creator,
    post_processor=None,
    verbose=1,
    key_names=Manager.Keys(elements='dataset'),
    save_method=bl.io.saver_dataset_triplet(bl.io.save_dataset),
    load_method=bl.io.loader_dataset_triplet(
        bl.io.add_bool_flag(bl.io.load_dataset, FileNotFoundError)
    ),
    table_saver=table_saver,
    table_loader=table_loader,
    description_comparer=description_comparer,
)

dataset_manager = Manager(
    path=analyses_path / 'datasets' / 'boiling_datasets',
    id_fmt='dataset {index}',
    index_key='index',
    creator=dataset_creator,
    post_processor=dataset_post_processor,
    verbose=1,
    key_names=Manager.Keys(elements='dataset'),
    table_saver=table_saver,
    table_loader=table_loader,
    description_comparer=description_comparer,
)

model_manager = Manager(
    path=analyses_path / 'models' / 'trained_models',
    id_fmt='{index}.model',
    index_key='index',
    save_method=bl.io.save_serialized(save_map),
    load_method=bl.io.add_bool_flag(bl.io.load_serialized(load_map), (FileNotFoundError, OSError)),
    verbose=2,
    key_names=Manager.Keys(elements='model'),
    table_saver=table_saver,
    table_loader=table_loader,
    description_comparer=description_comparer,
)

experiment_video_dataset_manager.verbose = 2
dataset_manager.verbose = 2

dataset_id, (ds_train, ds_val, ds_test) = make_dataset.main(
    experiment_video_dataset_manager=experiment_video_dataset_manager,
    dataset_manager=dataset_manager,
    img_ds=boiling_cases_timed()[1],
    splits=bl.datasets.DatasetSplits(
        train=Fraction(70, 100),
        val=Fraction(15, 100),
        test=Fraction(15, 100),
    ),
    preprocessors=boiling_preprocessors,
    augmentors=boiling_augmentors,
    augment_train=True,
    augment_test=True,
    # dataset_size=None,
    dataset_size=Fraction(1, 1000),
    shuffle=True,
    shuffle_size=None,
    batch_size=256,
    verbose=True,
)

assert isinstance(ds_train, SliceableDataset)
print("ds_train[0]:", ds_train[0])
print("ds_train[0][0]:", ds_train[0][0])

ds_train = sliceable_dataset_to_tensorflow_dataset(ds_train)
if ds_val is not None:
    ds_val = sliceable_dataset_to_tensorflow_dataset(ds_val)
ds_test = sliceable_dataset_to_tensorflow_dataset(ds_test)

ds_val_gt10 = bl.datasets.apply_unbatched(
    ds_val,
    # the following quantity is defined on
    # "Visualization-based nucleate boiling heat flux quantification
    # using machine learning"
    lambda ds: ds.take(12614),
    dim=0,
    key=0,
)
ds_val_gt10 = bl.datasets.filter_unbatched(
    ds_val_gt10, lambda img, data: data['Flux [W/cm**2]'] >= 10, dim=0, key=0
)

ds_contents = dataset_manager.contents(dataset_id)

model_id, model_dict = make_model.main(
    model_manager,
    strategy,
    dataset_contents_train=ds_contents,
    dataset_contents_val=ds_contents,
    ds_train=ds_train,
    ds_val=ds_val,
    take_train=None,
    take_val=None,
    additional_val_sets={'HF10': ds_val_gt10},
    model_creator=HoboldNet3,
    lr=1e-5,
    target='Flux [W/cm**2]',
    normalize_images=False,
    reduce_lr_on_plateau_factor=None,
    reduce_lr_on_plateau_patience=None,
    early_stopping_patience=10,
    dropout_ratio=0.5,
    batch_size=128,
    missing_ok=True,
    include=True,
    hidden_layers_policy='mixed_float16',
    output_layer_policy='float32',
)

print(model_dict)
