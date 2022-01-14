import os
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Container,
    Dict,
    ItemsView,
    KeysView,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    ValuesView,
)

import funcy
import json_tricks
import ray
import tensorflow as tf
import tensorflow_addons as tfa

import boiling_learning as bl
from boiling_learning.datasets.creators import (
    dataset_creator,
    dataset_post_processor,
    experiment_video_dataset_creator,
)
from boiling_learning.datasets.datasets import DatasetSplits
from boiling_learning.datasets.sliceable import (
    SliceableDataset,
    SupervisedSliceableDataset,
    concatenate,
    load_supervised_sliceable_dataset,
    save_supervised_sliceable_dataset,
    sliceable_dataset_to_tensorflow_dataset,
)
from boiling_learning.io import json
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.management.allocators.json_allocator import default_table_allocator
from boiling_learning.management.cacher import cache
from boiling_learning.management.managers import Manager
from boiling_learning.model.definitions import SmallConvNet
from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import Video, VideoFrame
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
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.descriptors import describe
from boiling_learning.utils.lazy import Lazy, LazyCallable
from boiling_learning.utils.typeutils import Many, typename
from boiling_learning.utils.utils import PathLike, print_header, resolve

ray.init()


print_header('Initializing script')


class Options(NamedTuple):
    login_user: bool = False
    convert_videos: bool = True
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
        pre_load_videos=OPTIONS.pre_load_videos,
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
        pre_load_videos=OPTIONS.pre_load_videos,
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

# boiling_preprocessors, boiling_augmentors = make_boiling_processors.main(
#     direct_visualization=True,
#     downscale_factor=5,
#     direct_height=180,
#     indirect_height=108,
#     indirect_height_ratio=0.4,
#     width=128,
# )
boiling_preprocessors, boiling_augmentors = make_boiling_processors.main(
    direct_visualization=True,
    downscale_factor=6,
    direct_height=90,
    indirect_height=108,
    indirect_height_ratio=0.4,
    width=64,
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


@describe.instance(Video)
def _describe_video(obj: Video) -> Dict[str, str]:
    return {'path': obj.path}


@describe.instance(ImageDataset)
def _describe_image_dataset(obj: ImageDataset) -> Dict[str, str]:
    return describe(list(obj))


@cache(
    allocator=default_table_allocator(analyses_path / 'datasets' / 'frames'),
    saver=bl.io.save_image,
    loader=bl.io.load_image,
)
def get_frame(
    video: Video,
    index: int,
    transformers: Sequence[Transformer[VideoFrame, VideoFrame]],
) -> VideoFrame:
    frame = video[index]
    for transformer in transformers:
        frame = transformer(frame)
    return frame


def sliceable_dataset_from_video_and_transformers(
    video: ExperimentVideo, transformers: Sequence[Transformer[VideoFrame, VideoFrame]]
) -> SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]:
    features = SliceableDataset.from_func(
        partial(get_frame, video=video, transformers=transformers), length=len(video)
    )
    targets = SliceableDataset(video.targets())
    return SupervisedSliceableDataset.from_features_and_targets(features, targets)


def _feature_saver(image: VideoFrame, path: PathLike) -> None:
    bl.io.save_image(image, resolve(path).with_suffix('.png'))


def _feature_loader(path: PathLike) -> None:
    return bl.io.load_image(resolve(path).with_suffix('.png'))


sliceable_dataset_saver = partial(
    save_supervised_sliceable_dataset, feature_saver=_feature_saver, target_saver=json.dump
)

sliceable_dataset_loader = partial(
    load_supervised_sliceable_dataset, feature_loader=_feature_loader, target_loader=json.load
)


@dataclass(frozen=True)
class GetImageDatasetParams:
    image_dataset: ImageDataset
    transformers: Sequence[Transformer[VideoFrame, VideoFrame]]
    splits: DatasetSplits
    dataset_size: Optional[Union[int, Fraction]] = None


def _get_image_dataset(
    image_dataset: ImageDataset,
    transformers: Sequence[Transformer[VideoFrame, VideoFrame]],
    splits: DatasetSplits,
    dataset_size: Optional[Union[int, Fraction]] = None,
) -> DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]]:
    ds = concatenate(
        sliceable_dataset_from_video_and_transformers(video, transformers)
        for video in image_dataset.values()
    )

    if dataset_size is not None:
        ds = ds.take(dataset_size)

    return ds.shuffle().split(splits.train, splits.val, splits.test)


@cache(
    allocator=default_table_allocator(analyses_path / 'datasets' / 'sliceable_image_datasets'),
    saver=bl.io.saver_dataset_triplet(sliceable_dataset_saver),
    loader=bl.io.loader_dataset_triplet(
        bl.io.add_bool_flag(sliceable_dataset_loader, FileNotFoundError)
    ),
)
def get_image_dataset(
    params: GetImageDatasetParams,
) -> DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]]:
    return _get_image_dataset(
        params.image_dataset, params.transformers, params.splits, params.dataset_size
    )


@dataclass(frozen=True)
class AugmentDatasetParams:
    augmentors: Sequence[Transformer[VideoFrame, VideoFrame]]
    batch_size: Optional[int] = None
    take: Optional[Union[int, Fraction]] = None
    augment_train: bool = True
    augment_test: bool = True
    augmentors_to_force: Container[str] = frozenset({'random_cropper'})


def apply_transformers_to_supervised_sliceable_dataset(
    dataset: SupervisedSliceableDataset[VideoFrame, Dict[str, Any]],
    augmentors: Sequence[
        Transformer[Tuple[VideoFrame, Dict[str, Any]], Tuple[VideoFrame, Dict[str, Any]]]
    ],
) -> SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]:
    for augmentor in augmentors:
        dataset = dataset.map(augmentor)
    return dataset


def _augment_datasets(
    datasets: DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]],
    augmentors: Sequence[Transformer[VideoFrame, VideoFrame]],
    batch_size: Optional[int] = None,
    take: Optional[Union[int, Fraction]] = None,
    augment_train: bool = True,
    augment_test: bool = True,
    augmentors_to_force: Container[str] = frozenset({'random_cropper'}),
) -> DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]]:
    ds_train, ds_val, ds_test = datasets
    if take is not None:
        ds_train = ds_train.take(take)
        if ds_val is not None:
            ds_val = ds_val.take(take)
        ds_test = ds_test.take(take)

    filtered_augmentors = (
        augmentors
        if augment_test
        else tuple(augmentor for augmentor in augmentors if augmentor.name in augmentors_to_force)
    )
    train_augmentors = augmentors if augment_train else filtered_augmentors
    test_augmentors = augmentors if augment_test else filtered_augmentors

    ds_train = apply_transformers_to_supervised_sliceable_dataset(ds_train, train_augmentors)
    ds_val = apply_transformers_to_supervised_sliceable_dataset(ds_val, test_augmentors)
    ds_test = apply_transformers_to_supervised_sliceable_dataset(ds_test, test_augmentors)

    ds_train = ds_train.shuffle()
    if ds_val is not None:
        ds_val = ds_val.shuffle()
    ds_test = ds_test.shuffle()

    if batch_size is not None:
        ds_train = ds_train.batch(batch_size)
        if ds_val is not None:
            ds_val = ds_val.batch(batch_size)
        ds_test = ds_test.batch(batch_size)

    return ds_train, ds_val, ds_test


def augment_datasets(
    datasets: DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]],
    params: AugmentDatasetParams,
) -> DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]]:
    return _augment_datasets(
        datasets,
        augmentors=params.augmentors,
        batch_size=params.batch_size,
        take=params.take,
        augment_train=params.augment_train,
        augment_test=params.augment_test,
        augmentors_to_force=params.augmentors_to_force,
    )


@dataclass(frozen=True)
class FitModelParams:
    architecture: tf.keras.Model
    strategy: tf.distribute.Strategy
    take_train: Optional[Union[int, Fraction]]
    take_val: Optional[Union[int, Fraction]]
    target: str
    additional_val_sets: Dict[str, tf.data.Dataset]
    lr: float
    normalize_images: bool
    reduce_lr_on_plateau_factor: float
    reduce_lr_on_plateau_patience: int
    early_stopping_patience: int
    dropout_ratio: float
    hidden_layers_policy: str
    output_layer_policy: str


@dataclass
class FittedModel:
    model: tf.keras.Model
    history: tf.keras.History


def _fit_model(
    augmented_datasets: DatasetTriplet[SupervisedSliceableDataset[VideoFrame, Dict[str, Any]]],
    architecture: tf.keras.Model,
    strategy: tf.distribute.Strategy,
    take_train: Optional[Union[int, Fraction]],
    take_val: Optional[Union[int, Fraction]],
    target: str,
    additional_val_sets: Dict[str, tf.data.Dataset],
    lr: float,
    normalize_images: bool,
    reduce_lr_on_plateau_factor: float,
    reduce_lr_on_plateau_patience: int,
    early_stopping_patience: int,
    dropout_ratio: float,
    hidden_layers_policy: str,
    output_layer_policy: str,
) -> FittedModel:
    # TODO: here!!!
    pass


@cache(
    allocator=default_table_allocator(analyses_path / 'models' / 'trained_models2'),
    saver=bl.io.saver_dataset_triplet(sliceable_dataset_saver),
    loader=bl.io.loader_dataset_triplet(
        bl.io.add_bool_flag(sliceable_dataset_loader, FileNotFoundError)
    ),
)
def fit_model(
    image_dataset_get_params: GetImageDatasetParams,
    image_dataset_augment_params: AugmentDatasetParams,
    fit_model_params: FitModelParams,
) -> FittedModel:
    return _fit_model(
        augment_datasets(
            get_image_dataset(image_dataset_get_params), image_dataset_augment_params
        ),
        architecture=fit_model_params.architecture,
        strategy=fit_model_params.strategy,
        take_train=fit_model_params.take_train,
        take_val=fit_model_params.take_val,
        target=fit_model_params.target,
        additional_val_sets=fit_model_params.additional_val_sets,
        lr=fit_model_params.lr,
        normalize_images=fit_model_params.normalize_images,
        reduce_lr_on_plateau_factor=fit_model_params.reduce_lr_on_plateau_factor,
        reduce_lr_on_plateau_patience=fit_model_params.reduce_lr_on_plateau_patience,
        early_stopping_patience=fit_model_params.early_stopping_patience,
        dropout_ratio=fit_model_params.dropout_ratio,
        hidden_layers_policy=fit_model_params.hidden_layers_policy,
        output_layer_policy=fit_model_params.output_layer_policy,
    )


experiment_video_dataset_manager = Manager(
    path=analyses_path / 'datasets' / 'boiling_experiment_video_datasets2',
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
    path=analyses_path / 'datasets' / 'boiling_datasets2',
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
    verbose=1,
    key_names=Manager.Keys(elements='model'),
    table_saver=table_saver,
    table_loader=table_loader,
    description_comparer=description_comparer,
)

experiment_video_dataset_manager.verbose = 2
dataset_manager.verbose = 2

image_dataset = boiling_cases_timed()[1]

print('Size of the input dataset:', sum(len(ev) for ev in image_dataset.values()))

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
    dataset_size=Fraction(1, 10),
    shuffle=True,
    shuffle_size=None,
    # batch_size=256,
    verbose=False,
)

print('Size of the TRAIN SET:', len(ds_train.flatten()), f'({len(ds_train)} batches)')
print('Size of the VAL SET:', len(ds_val.flatten()) if ds_val is not None else None)
print('Size of the TEST SET:', len(ds_test.flatten()), f'({len(ds_test)} batches)')

assert isinstance(ds_train, SliceableDataset), type(ds_train)
assert isinstance(ds_val, SliceableDataset), type(ds_val)
assert isinstance(ds_test, SliceableDataset), type(ds_test)
assert ds_train
assert ds_val
assert ds_test

ds_train = sliceable_dataset_to_tensorflow_dataset(ds_train).batch(16)
if ds_val is not None:
    ds_val = sliceable_dataset_to_tensorflow_dataset(ds_val).batch(16)
ds_test = sliceable_dataset_to_tensorflow_dataset(ds_test).batch(16)

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
    model_creator=SmallConvNet,
    lr=1e-5,
    target='Flux [W/cm**2]',
    normalize_images=False,
    reduce_lr_on_plateau_factor=None,
    reduce_lr_on_plateau_patience=None,
    early_stopping_patience=10,
    dropout_ratio=0.5,
    batch_size=16,
    missing_ok=True,
    include=True,
    hidden_layers_policy='mixed_float16',
    output_layer_policy='float32',
)

# TODO: try to convert the dataset type from float64 to float32 or float16 to reduce memory usage
# I have 8GB RAM whereas Google Colab has 12GB... not that much of a difference

print(model_dict)
