import itertools
import os
import sys
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache, partial
from operator import itemgetter
from pprint import pprint
from typing import (
    Any,
    Callable,
    ItemsView,
    Iterable,
    KeysView,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    ValuesView,
)

import funcy
import matplotlib.pyplot as plt
import modin.pandas as pd
import more_itertools as mit
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger
from rich.console import Console
from rich.table import Table
from skimage.io import imshow
from typing_extensions import ParamSpec

from boiling_learning.app.configuration import configure
from boiling_learning.automl.hypermodels import ConvImageRegressor, HyperModel
from boiling_learning.automl.tuners import EarlyStoppingGreedy
from boiling_learning.automl.tuning import TuneModelParams, TuneModelReturn, fit_hypermodel
from boiling_learning.datasets.bridging import sliceable_dataset_to_tensorflow_dataset
from boiling_learning.datasets.cache import EagerCache, NumpyCache
from boiling_learning.datasets.datasets import DatasetSplits, DatasetTriplet
from boiling_learning.datasets.sliceable import SliceableDataset, map_targets, targets
from boiling_learning.descriptions import describe
from boiling_learning.image_datasets import Image, ImageDataset, ImageDatasetTriplet, Targets
from boiling_learning.io.storage import dataclass
from boiling_learning.lazy import Lazy, LazyDescribed
from boiling_learning.management.allocators import JSONTableAllocator
from boiling_learning.management.cacher import CachedFunction, Cacher, cache
from boiling_learning.model.callbacks import (
    AdditionalValidationSets,
    MemoryCleanUp,
    RegisterEpoch,
    SaveHistory,
    TimePrinter,
)
from boiling_learning.model.definitions import hoboldnet2
from boiling_learning.model.model import ModelArchitecture, rename_model_layers
from boiling_learning.model.training import (
    CompiledModel,
    CompileModelParams,
    FitModelParams,
    FitModelReturn,
    compile_model,
    get_fit_model,
    load_with_strategy,
    strategy_scope,
)
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.scripts import (
    connect_gpus,
    load_cases,
    load_dataset_tree,
    make_boiling_processors,
    make_condensation_processors,
    set_boiling_cases_data,
    set_condensation_datasets_data,
)
from boiling_learning.scripts.utils.initialization import check_all_paths_exist
from boiling_learning.transforms import dataset_sampler, datasets_merger, map_transformers, subset
from boiling_learning.utils.functional import P
from boiling_learning.utils.pathutils import resolve
from boiling_learning.utils.typeutils import typename

# TODO: check <https://stackoverflow.com/a/58970598/5811400> and <https://github.com/googlecolab/colabtools/issues/864#issuecomment-556437040>
# TODO: na condensação, fazer crop determinístico!!!
# TODO: depois, se tiver tempo, fazer RandomCrop pra comparar
# TODO: ver o quanto influencia crop determinístico versus randomico
# TODO: para ambos os tipos de corte, rodar auto ML
# TODO: para um mesmo fluxo, mostrar imagens para os quatro datasets
# TODO: fazer vídeos tipo o do Hobold com a ebulição e um gráfico (barrinha de erro), o fluxo nominal e o valor predito
# TODO: esse vídeo pode ser para as quatro superfícies ao mesmo tempo


configure(
    force_gpu_allow_growth=True,
    use_xla=True,
    mixed_precision_global_policy='mixed_float16',
    modin_engine='ray',
)


masters_path = resolve(os.environ['MASTERS_PATH'])
data_path = masters_path / 'data'
boiling_data_path = data_path / 'boiling1d'
boiling_experiments_path = boiling_data_path / 'experiments'
boiling_cases_path = boiling_data_path / 'cases'
condensation_data_path = data_path / 'condensation'
analyses_path = masters_path / 'analyses'
tensorboard_logs_path = resolve(analyses_path / 'models' / 'logs', dir=True)

log_file = resolve(masters_path / 'logs' / '{time}.log', parents=True)

logger.remove()
logger.add(sys.stderr, level='DEBUG')
logger.add(str(log_file), level='DEBUG')

logger.info('Initializing script')


class Options(NamedTuple):
    test: bool = False
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


OPTIONS = Options()
logger.info(f'Options: {OPTIONS}')

logger.info('Checking paths')
check_all_paths_exist(
    (
        ('Masters', masters_path),
        ('Boiling data', boiling_data_path),
        ('Boiling cases', boiling_cases_path),
        ('Boiling experiments', boiling_experiments_path),
        ('Contensation data', condensation_data_path),
        ('Analyses', analyses_path),
    )
)
logger.info('Succesfully checked paths')

strategy = connect_gpus.main(require_gpu=False)
strategy = LazyDescribed.from_value_and_description(strategy, typename(strategy))
logger.info(f'Using distribute strategy: {describe(strategy)}')

"""## Datasets

### Image datasets
"""

boiling_cases_names = tuple(f'case {idx+1}' for idx in range(5))
boiling_cases_names_timed = tuple(funcy.without(boiling_cases_names, 'case 1'))

logger.info('Preparing datasets')
logger.info('Loading cases')
logger.info(f'Loading boiling cases from {boiling_cases_path}')
boiling_cases = load_cases.main(
    (boiling_cases_path / case_name for case_name in boiling_cases_names),
    video_suffix='.MP4',
    convert_videos=OPTIONS.convert_videos,
)
boiling_cases_timed = tuple(
    case for case in boiling_cases if case.name in boiling_cases_names_timed
)

boiling_experiments_map = {
    'case 1': boiling_experiments_path / 'Experiment 2020-08-03 16-19' / 'data.csv',
    'case 2': boiling_experiments_path / 'Experiment 2020-08-05 14-15' / 'data.csv',
    'case 3': boiling_experiments_path / 'Experiment 2020-08-05 17-02' / 'data.csv',
    'case 4': boiling_experiments_path / 'Experiment 2020-08-28 15-28' / 'data.csv',
    'case 5': boiling_experiments_path / 'Experiment 2020-09-10 13-53' / 'data.csv',
}

logger.info(f'Loading condensation cases from {condensation_data_path}')
condensation_datasets = load_dataset_tree.main(condensation_data_path)
condensation_data_spec_path = condensation_data_path / 'data_spec.yaml'

BOILING_VIDEO_TO_SETTER = {
    video.name: partial(
        set_boiling_cases_data.main,
        boiling_cases_timed,
        case_experiment_map=boiling_experiments_map,
    )
    for case in boiling_cases_timed
    for video in case
}

CONDENSATION_VIDEO_TO_SETTER = {
    video.name: partial(
        set_condensation_datasets_data.main, condensation_datasets, condensation_data_spec_path
    )
    for img_ds in condensation_datasets
    for video in img_ds
}

VIDEO_TO_SETTER = {**BOILING_VIDEO_TO_SETTER, **CONDENSATION_VIDEO_TO_SETTER}


def ensure_data_is_set(video: ExperimentVideo) -> None:
    if video.data is None:
        setter = VIDEO_TO_SETTER[video.name]
        setter()


@cache(JSONTableAllocator(analyses_path / 'cache' / 'purged-experiment-videos'))
def purge_experiment_videos(image_dataset: ExperimentVideoDataset) -> list[str]:
    for video in tuple(image_dataset):
        ensure_data_is_set(video)
        # assert video.data is not None or video not in image_dataset, (video.name, image_dataset)

    return [video.name for video in image_dataset if video.data is not None]


# TODO: accept parameters directly in `get_image_dataset`
# and stop using `GetImageDatasetParams` - but keep it internally
# to avoid changing the description


numpy_directory_allocator = JSONTableAllocator(analyses_path / 'datasets' / 'numpy', suffix='')
targets_allocator = JSONTableAllocator(analyses_path / 'datasets' / 'targets', suffix='.csv')


def _compile_transformers(
    transformers: Iterable[Transformer[Image, Image]],
    experiment_video: ExperimentVideo,
) -> LazyDescribed[Callable[[Image], Image]]:
    compiled_transformers = tuple(
        transformer[experiment_video.name] if isinstance(transformer, dict) else transformer
        for transformer in transformers
    )
    return LazyDescribed.from_value_and_description(
        funcy.rcompose(*compiled_transformers), compiled_transformers
    )


@dataclass
class VideoInfo:
    length: int
    shape: tuple[int, ...]
    dtype: str


@cache(JSONTableAllocator(analyses_path / 'cache' / 'video-info'))
def get_video_info(video: Lazy[SliceableDataset[Image]]) -> VideoInfo:
    dataset = video()
    first_frame = dataset[0]
    return VideoInfo(
        length=len(dataset),
        shape=first_frame.shape,
        dtype=str(first_frame.dtype),
    )


EAGER_BUFFER_SIZE = 32


def _video_dataset_from_video_and_transformers(
    experiment_video: ExperimentVideo,
    transformers: Iterable[Transformer[Image, Image]],
) -> SliceableDataset[Image]:
    compiled_transformers = _compile_transformers(transformers, experiment_video)

    video = LazyDescribed.from_value_and_description(
        experiment_video.video, experiment_video
    ) | map_transformers(compiled_transformers)
    video_info = get_video_info(video)

    directory = numpy_directory_allocator.allocate(video)
    numpy_cache = NumpyCache(
        directory,
        shape=(video_info.length, *video_info.shape),
        dtype=np.dtype(video_info.dtype),
    )
    return video().cache(EagerCache(numpy_cache, buffer_size=EAGER_BUFFER_SIZE))


@lru_cache(maxsize=1024)
def _target_dataset_from_video(video: ExperimentVideo) -> SliceableDataset[Targets]:
    path = targets_allocator.allocate(video)

    try:
        targets = pd.read_csv(path)
    except (OSError, AttributeError):
        ensure_data_is_set(video)
        targets = video.targets()
        targets.to_csv(path, index=False)

    return SliceableDataset.from_sequence(targets.to_dict('records'))


def sliceable_dataset_from_video_and_transformers(
    ev: ExperimentVideo,
    transformers: Iterable[Transformer[Image, Image]],
) -> ImageDataset:
    targets = _target_dataset_from_video(ev)  # targets first to ensure correct ev data setup
    video = _video_dataset_from_video_and_transformers(ev, transformers)

    return SliceableDataset.zip(video, targets, strictness='one-off')
    # return SliceableDataset.zip(video, targets, strictness="none")
    # return SliceableDataset.zip(video, targets, strictness="none" if ev.name in CONDENSATION_VIDEO_TO_SETTER else "one-off")


if OPTIONS.test:
    sample_ev = mit.first(boiling_cases_timed[0])
    sds = sliceable_dataset_from_video_and_transformers(sample_ev, boiling_direct_preprocessors)
    sample_frame = sds[0][0]
    print(sample_frame)


@dataclass(frozen=True)
class GetImageDatasetParams:
    image_dataset: ExperimentVideoDataset
    transformers: list[Transformer[Image, Image]]
    splits: DatasetSplits = DatasetSplits(
        train=Fraction(70, 100),
        val=Fraction(15, 100),
        test=Fraction(15, 100),
    )


def _get_image_dataset(
    image_dataset: ExperimentVideoDataset,
    transformers: list[Transformer[Image, Image]],
    splits: DatasetSplits,
) -> ImageDatasetTriplet:
    purged_experiment_videos = purge_experiment_videos(image_dataset)

    ds_train_list = []
    ds_val_list = []
    ds_test_list = []
    for video in sorted(image_dataset, key=lambda ev: ev.name):
        if video.name in purged_experiment_videos:
            dataset = sliceable_dataset_from_video_and_transformers(video, transformers)
            ev_train, ev_val, ev_test = dataset.split(splits.train, splits.val, splits.test)

            ds_train_list.append(ev_train)
            ds_val_list.append(ev_val)
            ds_test_list.append(ev_test)

    # TODO: re-add memory caching here
    ds_train = SliceableDataset.concatenate(*ds_train_list)  # .cache(MemoryCache())
    ds_val = SliceableDataset.concatenate(*ds_val_list)  # .cache(MemoryCache())
    ds_test = SliceableDataset.concatenate(*ds_test_list)  # .cache(MemoryCache())

    return ds_train, ds_val, ds_test


def get_image_dataset(
    params: GetImageDatasetParams,
) -> LazyDescribed[ImageDatasetTriplet]:
    return LazyDescribed.from_value_and_description(
        _get_image_dataset(params.image_dataset, params.transformers, params.splits), params
    )


if OPTIONS.test:
    get_image_dataset_params = GetImageDatasetParams(
        boiling_cases_timed[0], transformers=boiling_direct_preprocessors
    )
    ds_train, ds_val, ds_test = get_image_dataset(get_image_dataset_params)()
    ds_train_len = len(ds_train)
    ds_val_len = len(ds_val)
    ds_test_len = len(ds_test)
    expected_length = sum(len(ev) for ev in boiling_cases_timed[0])
    assert ds_train_len > ds_test_len > 0
    assert (
        ds_train_len + ds_val_len + ds_test_len == expected_length
    ), f'{ds_train_len} + {ds_val_len} + {ds_test_len} == {ds_train_len + ds_val_len + ds_test_len} != {expected_length}'
    sample_element = ds_train[0]
    assert isinstance(sample_element[0], np.ndarray)
    assert isinstance(sample_element[1], float)

"""### Sliceable datasets"""


_T = TypeVar('_T')

# DEFAULT_PREFETCH_BUFFER_SIZE = 4
DEFAULT_PREFETCH_BUFFER_SIZE = 1024

training_datasets_allocator = JSONTableAllocator(analyses_path / 'datasets' / 'training')


def _default_filter_for_frames_dataset(
    dataset: ImageDataset,
) -> Callable[[tuple[VideoFrame, dict[str, Any]]], bool]:
    def _pred(pair: tuple[VideoFrame, dict[str, Any]]) -> bool:
        if len(pair) != 2:
            return False

        first_frame, _ = dataset[0]
        frame, _data = pair
        return frame.shape == first_frame.shape and not np.allclose(frame, 0)

    return _pred


def to_tensorflow(
    dataset: LazyDescribed[ImageDataset],
    *,
    batch_size: Optional[int] = None,
    prefilterer: Optional[
        LazyDescribed[Callable[[tuple[VideoFrame, dict[str, Any]]], bool]]
    ] = None,
    filterer: Optional[Callable[..., bool]] = None,
    buffer_size: int = DEFAULT_PREFETCH_BUFFER_SIZE,
    target: Optional[str] = None,
    shuffle: Union[bool, int] = True,
) -> LazyDescribed[tf.data.Dataset]:
    dataset_value = dataset()

    default_prefilterer = _default_filter_for_frames_dataset(dataset_value)

    def _prefilterer(element: tuple[VideoFrame, dict[str, Any]]) -> bool:
        return default_prefilterer(element) and (prefilterer is None or prefilterer()(element))

    save_path = training_datasets_allocator.allocate(dataset, prefilterer)
    logger.debug(f'Converting dataset to TF and saving to {save_path}')

    if shuffle is True:
        dataset_value = dataset_value.shuffle()

    tf_dataset = sliceable_dataset_to_tensorflow_dataset(
        dataset_value,
        # DEBUG: I commented out the following line to avoid issues with dataset saving taking too long
        save_path=save_path,
        # DEBUG: try re-setting this to True
        cache=False,
        batch_size=batch_size,
        prefilterer=_prefilterer,
        filterer=filterer,
        prefetch=buffer_size,
        expand_to_batch_size=True,
        deterministic=False,
        target=target,
    )

    if shuffle and shuffle is not True:
        tf_dataset = tf_dataset.shuffle(shuffle)

    return LazyDescribed.from_value_and_description(
        tf_dataset,
        (
            dataset,
            ('prefilterer', prefilterer),
            ('batch', batch_size),
            ('target', target),
        ),
    )


def to_tensorflow_triplet(
    dataset: LazyDescribed[ImageDatasetTriplet],
    *,
    batch_size: Optional[int] = None,
    prefilterer: Optional[LazyDescribed[Callable[[_T], bool]]] = None,
    filterer: Optional[Callable[..., bool]] = None,
    buffer_size: int = DEFAULT_PREFETCH_BUFFER_SIZE,
    target: Optional[str] = None,
    include_train: bool = True,
    include_val: bool = True,
    include_test: bool = True,
    shuffle: Union[bool, int] = True,
) -> DatasetTriplet[LazyDescribed[tf.data.Dataset]]:
    _to_tensorflow = partial(
        to_tensorflow,
        batch_size=batch_size,
        prefilterer=prefilterer,
        filterer=filterer,
        buffer_size=buffer_size,
        target=target,
        shuffle=shuffle,
    )

    if include_train:
        logger.debug('Converting TRAIN set to tensorflow')
        ds_train = _to_tensorflow(dataset | subset('train'))
    else:
        ds_train = LazyDescribed.from_value_and_description(tf.data.Dataset.range(0), None)

    if include_val:
        logger.debug('Converting VAL set to tensorflow')
        ds_val = _to_tensorflow(dataset | subset('val'))
    else:
        ds_val = LazyDescribed.from_value_and_description(tf.data.Dataset.range(0), None)

    if include_test:
        logger.debug('Converting TEST set to tensorflow')
        ds_test = _to_tensorflow(dataset | subset('test'))
    else:
        ds_test = LazyDescribed.from_value_and_description(tf.data.Dataset.range(0), None)

    return DatasetTriplet(ds_train, ds_val, ds_test)


"""### Default datasets

#### On-wire pool boiling
"""


boiling_direct_preprocessors = make_boiling_processors.main(direct_visualization=True)
boiling_indirect_preprocessors = make_boiling_processors.main(direct_visualization=False)

# logger.debug("Displaying directly visualized boiling frames")

# TOTAL_EXAMPLES = sum(1 for case in boiling_cases_timed for ev in case)
# N_COLS = 4
# N_ROWS = math.ceil(TOTAL_EXAMPLES / N_COLS)

# fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS*8, N_ROWS*8))

# index = 0
# for case in boiling_cases_timed:
#     for ev in sorted(case, key=lambda ev: ev.name):
#         try:
#             transformer = _compile_transformers(boiling_direct_preprocessors, ev)
#         except KeyError:
#             # some experiment videos have no transformers associated
#             continue

#         logger.debug(f"Getting frame #0 from case {case.name} and video {ev.name}")
#         frame = ev.video[0]
#         logger.debug("Transforming frame")
#         frame = transformer()(frame)

#         logger.debug("Showing frame")
#         col = index % N_COLS
#         row = index // N_COLS

#         axs[row, col].imshow(frame.squeeze(), cmap="gray")
#         axs[row, col].set_title(f"{case.name} - {ev.name}")
#         axs[row, col].grid(False)

#         index += 1

# logger.debug("Showing figure")

# fig.show()

# logger.debug("Done")

# logger.debug("Displaying indirectly visualized boiling frames")

# TOTAL_EXAMPLES = sum(1 for case in boiling_cases_timed for ev in case)
# N_COLS = 4
# N_ROWS = math.ceil(TOTAL_EXAMPLES / N_COLS)

# fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS*8, N_ROWS*8))

# index = 0
# for case in boiling_cases_timed:
#     for ev in sorted(case, key=lambda ev: ev.name):
#         try:
#             transformer = _compile_transformers(boiling_indirect_preprocessors, ev)
#         except KeyError:
#             # some experiment videos have no transformers associated
#             continue

#         frame = transformer()(ev.video[0])

#         col = index % N_COLS
#         row = index // N_COLS

#         axs[row, col].imshow(frame.squeeze(), cmap="gray")
#         axs[row, col].set_title(f"{case.name} - {ev.name}")
#         axs[row, col].grid(False)

#         index += 1

# logger.debug("Showing figure")

# fig.show()

# logger.debug("Done")

logger.debug('Getting datasets')

boiling_direct_datasets = tuple(
    get_image_dataset(GetImageDatasetParams(case, transformers=boiling_direct_preprocessors))
    for case in boiling_cases_timed
)

boiling_indirect_datasets = tuple(
    get_image_dataset(GetImageDatasetParams(case, transformers=boiling_indirect_preprocessors))
    for case in boiling_cases_timed
)

# for is_direct, datasets in (
#     (True, boiling_direct_datasets),
#     (False, boiling_indirect_datasets),
# ):
#     for index, dataset in enumerate(datasets):
#         for subset_name, subset in zip(('train', 'val', 'test'), dataset()):
#             logger.info(
#                 f"Iterating over {'direct' if is_direct else 'indirect'} {subset_name} "
#                 f'dataset #{index}.'
#             )
#             for frame, targets in subset:
#                 pass

# logger.debug("Done")

# for index, dataset in enumerate(boiling_direct_datasets):
#     for subset_name, subset in zip(("train", "val", "test"), dataset()):
#         path = analyses_path / "outputs" / "animations" / f"boiling-{index}-direct-{subset_name}.mp4"

#         if not path.is_file():
#             save_as_video(
#                 path,
#                 subset[::60].prefetch(256),
#                 display_data={
#                     "index": "Index",
#                     "Flux [W/cm**2]": "Flux [W/cm²]"
#                 },
#                 fps=30
#             )

"""#### Condensation"""

# TODO: choose this correctly!!
CONDENSATION_SUBSAMPLE = Fraction(1, 60)

condensation_preprocessors = make_condensation_processors.main(
    downscale_factor=5, height=8 * 12, width=8 * 12
)
condensation_dataset = (
    tuple(
        get_image_dataset(
            GetImageDatasetParams(
                ds,
                condensation_preprocessors,
            )
        )
        | dataset_sampler(CONDENSATION_SUBSAMPLE)
        for ds in condensation_datasets
    )
    | datasets_concatenater()
)


# TODO: improve this!!!

"""## Models

### Training
"""

# DON'T FORGET TO TURN OFF SHIELDS: https://github.com/tensorflow/tensorboard/issues/3186#issuecomment-663534468


_P = ParamSpec('_P')


class FitBoilingModel(CachedFunction[_P, FitModelReturn]):
    def __init__(self, cacher: Cacher[FitModelReturn]) -> None:
        super().__init__(get_fit_model, cacher)

    def __call__(
        self,
        compiled_model: CompiledModel,
        datasets: LazyDescribed[ImageDatasetTriplet],
        params: FitModelParams,
        try_id: int = 0,
        target: str = 'Flux [W/cm**2]',
    ) -> FitModelReturn:
        """
        try_id: use this to force this model to be trained again. This may be used for instance to get a average and
            stddev.
        """

        def _is_not_outlier(pair: tuple[Image, Targets]) -> bool:
            frame, data = pair
            return abs(data['Power [W]'] - data['nominal_power']) < 5

        def _is_gt10(frame: Image, data: Targets) -> bool:
            return data[target] >= 10

        ds_val_g10 = to_tensorflow(
            datasets | subset('val'),
            prefilterer=LazyDescribed.from_value_and_description(
                _is_not_outlier, 'abs(Power [W] - nominal_power) < 5'
            ),
            filterer=_is_gt10,
            batch_size=params.batch_size,
            target=target,
        )

        params.callbacks().extend(
            (
                TimePrinter(
                    when={
                        'on_epoch_begin',
                        'on_epoch_end',
                        'on_predict_begin',
                        'on_predict_end',
                        'on_test_begin',
                        'on_test_end',
                        'on_train_begin',
                        'on_train_end',
                    }
                ),
                # BackupAndRestore(workspace_path / 'backup', delete_on_end=False),
                AdditionalValidationSets({'HF10': ds_val_g10()}),
                MemoryCleanUp(),
                # tf.keras.callbacks.TensorBoard(tensorboard_logs_path / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1),
            )
        )

        workspace_path = resolve(
            self.allocate(compiled_model, datasets, params, target, try_id), parents=True
        )

        creator: Callable[[], FitModelReturn] = P(
            compiled_model,
            tuple(
                subset()
                for subset in to_tensorflow_triplet(
                    datasets,
                    prefilterer=LazyDescribed.from_value_and_description(
                        _is_not_outlier, 'abs(Power [W] - nominal_power) < 5'
                    ),
                    batch_size=params.batch_size,
                    include_test=False,
                    target=target,
                )
            ),
            params,
            epoch_registry=RegisterEpoch(workspace_path / 'epoch.json'),
            history_registry=SaveHistory(workspace_path / 'history.json', mode='a'),
        ).partial(self.function)

        return self.provide(creator, workspace_path / 'model')


class FitCondensationModel(CachedFunction[_P, FitModelReturn]):
    def __init__(self, cacher: Cacher[FitModelReturn]) -> None:
        super().__init__(get_fit_model, cacher)

    def __call__(
        self,
        compiled_model: CompiledModel,
        datasets: LazyDescribed[ImageDatasetTriplet],
        params: FitModelParams,
        target: str,
    ) -> FitModelReturn:
        params.callbacks().extend(
            (
                TimePrinter(
                    when={
                        'on_epoch_begin',
                        'on_epoch_end',
                        'on_predict_begin',
                        'on_predict_end',
                        'on_test_begin',
                        'on_test_end',
                        'on_train_begin',
                        'on_train_end',
                    }
                ),
                # BackupAndRestore(workspace_path / 'backup', delete_on_end=False),
                MemoryCleanUp()
                # tf.keras.callbacks.TensorBoard(tensorboard_logs_path / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1),
            )
        )

        workspace_path = resolve(
            self.allocate(compiled_model, datasets, params, target), parents=True
        )

        creator: Callable[[], FitModelReturn] = P(
            compiled_model,
            to_tensorflow_triplet(
                datasets,
                batch_size=params.batch_size,
                include_test=False,
                target=target,
            ),
            params,
            epoch_registry=RegisterEpoch(workspace_path / 'epoch.json'),
            history_registry=SaveHistory(workspace_path / 'history.json', mode='a'),
        ).partial(self.function)

        return self.provide(creator, workspace_path / 'model')


fit_boiling_model = FitBoilingModel(
    Cacher(
        allocator=JSONTableAllocator(analyses_path / 'models' / 'boiling', suffix=''),
        exceptions=(FileNotFoundError, NotADirectoryError, tf.errors.OpError),
        loader=load_with_strategy(strategy),
    )
)

fit_condensation_model = FitCondensationModel(
    Cacher(
        allocator=JSONTableAllocator(analyses_path / 'models' / 'condensation', suffix=''),
        exceptions=(FileNotFoundError, NotADirectoryError, tf.errors.OpError),
        loader=load_with_strategy(strategy),
    )
)


if OPTIONS.test:
    get_image_dataset_params = GetImageDatasetParams(
        boiling_cases_timed[0],
        transformers=boiling_direct_preprocessors,
        dataset_size=None,
    )

    logger.info(f'Getting datasets...')
    ds_train, ds_val, ds_test = get_image_dataset(get_image_dataset_params)
    logger.info(f'Done')

    frame, _ = ds_train[0]

    imshow(frame.squeeze())

"""### Autofitting"""


boiling_direct_hypermodel_allocator = JSONTableAllocator(
    analyses_path / 'autofit' / 'boiling-direct-tuners'
)
boiling_indirect_hypermodel_allocator = JSONTableAllocator(
    analyses_path / 'autofit' / 'boiling-indirect-tuners'
)


@cache(
    allocator=JSONTableAllocator(analyses_path / 'autofit' / 'models'),
    exceptions=(FileNotFoundError, NotADirectoryError, tf.errors.OpError),
    loader=load_with_strategy(strategy),
)
def autofit(
    hypermodel: HyperModel,
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: TuneModelParams,
    target: str,
) -> TuneModelReturn:
    ds_train, ds_val, ds_test = to_tensorflow_triplet(
        datasets,
        batch_size=params.batch_size,
        include_test=False,
        target=target,
    )

    if not any(isinstance(callback, MemoryCleanUp) for callback in params.callbacks()):
        params.callbacks().append(MemoryCleanUp())

    tuned_model = fit_hypermodel(
        hypermodel,
        DatasetTriplet(
            ds_train().unbatch().prefetch(tf.data.AUTOTUNE),
            ds_val().unbatch().prefetch(tf.data.AUTOTUNE),
            ds_test(),
        ),
        params,
    )

    tuned_model = replace(tuned_model, model=rename_model_layers(tuned_model.model))

    return tuned_model


"""### Baseline on-wire pool boiling"""

logger.info('Getting sample frames')

baseline_boiling_dataset_direct = boiling_direct_datasets[0]
baseline_boiling_dataset_indirect = boiling_indirect_datasets[0]

ds_train_direct, _, _ = baseline_boiling_dataset_direct()
first_frame_direct, _ = ds_train_direct[0]
ds_train_indirect, _, _ = baseline_boiling_dataset_indirect()
first_frame_indirect, _ = ds_train_indirect[0]

baseline_boiling_mse_direct = 13  # W / cm**2
baseline_boiling_mse_indirect = 33  # W / cm**2

logger.info('Done getting sample frames')


def get_baseline_compile_params() -> CompileModelParams:
    with strategy_scope(strategy):
        return CompileModelParams(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=[
                tf.keras.metrics.MeanSquaredError('MSE'),
                tf.keras.metrics.RootMeanSquaredError('RMS'),
                tf.keras.metrics.MeanAbsoluteError('MAE'),
                tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
                tfa.metrics.RSquare('R2'),
            ],
        )


def get_baseline_boiling_model(
    direct: bool = True, normalize_images: bool = True
) -> ModelArchitecture:
    first_frame = first_frame_direct if direct else first_frame_indirect

    with strategy_scope(strategy):
        return hoboldnet2(first_frame.shape, dropout=0.5, normalize_images=normalize_images)


logger.info('Getting direct baseline model')
baseline_boiling_model_architecture_direct = get_baseline_boiling_model(
    direct=True, normalize_images=False
)
logger.info('Done getting direct baseline model')

logger.info('Getting indirect baseline model')
baseline_boiling_model_architecture_indirect = get_baseline_boiling_model(
    direct=False, normalize_images=False
)
logger.info('Done getting indirect baseline model')

baseline_boiling_model_direct_size = int(
    baseline_boiling_model_architecture_direct.count_parameters(
        trainable=True,
        non_trainable=False,
    )
)

baseline_boiling_model_indirect_size = int(
    baseline_boiling_model_architecture_indirect.count_parameters(
        trainable=True,
        non_trainable=False,
    )
)


def get_baseline_fit_params() -> FitModelParams:
    return FitModelParams(
        batch_size=200,
        epochs=100,
        callbacks=LazyDescribed.from_list(
            [
                LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
                LazyDescribed.from_constructor(
                    tf.keras.callbacks.EarlyStopping,
                    monitor='val_loss',
                    min_delta=0,
                    patience=10,
                    baseline=None,
                    mode='auto',
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
        ),
    )


def get_pretrained_baseline_boiling_model(
    direct: bool = True, normalize_images: bool = True
) -> FitModelReturn:
    compiled_model = compile_model(
        get_baseline_boiling_model(direct=direct, normalize_images=normalize_images),
        get_baseline_compile_params(),
    )

    return fit_boiling_model(
        compiled_model,
        baseline_boiling_dataset_direct if direct else baseline_boiling_dataset_indirect,
        get_baseline_fit_params(),
        target='Flux [W/cm**2]',
    )


# pretrained_baseline_boiling_model_architecture_direct = get_pretrained_baseline_boiling_model(direct=True, normalize_images=False)
# pretrained_baseline_boiling_model_architecture_indirect = get_pretrained_baseline_boiling_model(direct=False, normalize_images=False)


_autofit_to_dataset_allocator = JSONTableAllocator(
    analyses_path / 'autofit' / 'autofit-to-dataset'
)


def autofit_to_dataset(
    datasets: LazyDescribed[ImageDatasetTriplet],
    *,
    target: str,
    normalize_images: bool = True,
    max_model_size: Optional[int] = None,
    goal: Optional[float] = None,
) -> TuneModelReturn:
    compile_params = get_baseline_compile_params()

    hypermodel = ConvImageRegressor(
        loss=compile_params.loss,
        metrics=compile_params.metrics,
        tuner=EarlyStoppingGreedy,
        directory=_autofit_to_dataset_allocator.allocate(
            ConvImageRegressor,
            datasets,
            tuner=EarlyStoppingGreedy,
            loss=compile_params.loss,
            metrics=compile_params.metrics,
            normalize_images=normalize_images,
            max_model_size=max_model_size,
            goal=goal,
        ),
        max_model_size=max_model_size,
        strategy=strategy,
        normalize_images=normalize_images,
        goal=goal,
    )

    tune_model_params = TuneModelParams(
        batch_size=32,
        callbacks=LazyDescribed.from_list(
            [
                LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
                LazyDescribed.from_constructor(
                    tf.keras.callbacks.EarlyStopping,
                    monitor='val_loss',
                    min_delta=0,
                    patience=10,
                    baseline=None,
                    mode='auto',
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
        ),
    )

    return autofit(
        hypermodel,
        datasets=datasets,
        params=tune_model_params,
        target=target,
    )


"""## Pre-processing analyses"""

# TODO: generate a test case in which temperature is estimated from the boiling curve and that's what the models have to predict
# TODO: data clean up; remove data from experiments where measured heatflux is 5W/cm2 or more away from its level
# TODO: fazer erro em função do y: ver se para maiores ys o erro vai subindo ou diminuindo
# quem sabe fazer 3 ou mais modelos, um especializado para cada região de y; e quem sabe
# usar um classificador pra escolher qual estimador não ajude muito
# focar na arquitetura da rede, que é mais importante do que hiperparâmetros
# otimizar as convolucionais pode ser mais importante do que otimizar as fully-connected

# TODO: ReLU or LeakyReLU? https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning

# TODO: hypothesis: it is better to _not_ normalize images.
# Does this mean that the model uses the overall image brightness to do its inference?
# If so, this is bad.
# This can be assessed by training two models (one with normalized images and the other without) and comparing the
# relative importance that each one of them gives to the areas without bubbles

# TODO: does the model tend to overestimate or underestimate values?
# TODO: train the same network multiple times to get an average and stddev of the error. I noticed that, by training the
# same model multiple times, I got R2 scores of ~0.96, 0.90 and 0.94. Hobold got 0.98... Maybe it's just because he
# tried a lot of times until he got a good performance?

"""### Data Distribution"""


PREFETCH = 2048


@cache(JSONTableAllocator(analyses_path / 'cache' / 'targets'))
def get_targets(
    dataset: LazyDescribed[ImageDatasetTriplet],
) -> tuple[list[Targets], list[Targets], list[Targets]]:
    ds_train, ds_val, ds_test = dataset()

    return (
        list(targets(ds_train).prefetch(PREFETCH)),
        list(targets(ds_val).prefetch(PREFETCH)),
        list(targets(ds_test).prefetch(PREFETCH)),
    )


def plot_dataset_targets(
    datasets: tuple[LazyDescribed[ImageDatasetTriplet], ...],
    *,
    target_name: str,
    filter_target: Optional[Callable[[Targets], bool]] = None,
) -> None:
    sns.set_style('whitegrid')

    f, axes = plt.subplots(len(datasets), 3, figsize=(9, 9), sharey='row')

    for row, splits in enumerate(datasets):
        targets_train, targets_val, targets_test = get_targets(splits)
        for col, (title, ys) in enumerate(
            (
                ('train', targets_train),
                ('val', targets_val),
                ('test', targets_test),
            )
        ):
            y = [
                target[target_name]
                for target in ys
                if filter_target is None or filter_target(target)
            ]
            x = range(len(y))
            axes[row, col].scatter(x, y)

            if not row:
                axes[row, col].set_title(title)

            if not col:
                axes[row, col].set_ylabel(f'Dataset #{row + 1}')


"""#### On-wire pool boiling"""


boiling_filter_target = lambda target: abs(target['Power [W]'] - target['nominal_power']) < 5
boiling_target_name = 'Flux [W/cm**2]'

# plot_dataset_targets(
#     boiling_direct_datasets,
#     target_name=boiling_target_name,
#     filter_target=boiling_filter_target
# )

# plot_dataset_targets(
#     boiling_indirect_datasets,
#     target_name=boiling_target_name,
#     filter_target=boiling_filter_target
# )

"""#### Condensation"""

# TODO: ensure that this works!
# plot_dataset_targets(condensation_datasets)

"""### Downscaling"""

# from itertools import takewhile
#

# from boiling_learning.preprocessing.image import (
#     Downscaler,
#     normalized_mutual_information,
#     retained_variance,
#     shannon_cross_entropy_ratio,
#     shannon_entropy_ratio,
#     structural_similarity_ratio
# )

# preprocessors = list(
#     takewhile(
#         lambda preprocessor: not isinstance(preprocessor, Downscaler),
#         boiling_direct_preprocessors
#     )
# )

# sample_frames: list[VideoFrame] = []
# for case in boiling_cases_timed:
#     get_image_dataset_params = GetImageDatasetParams(
#         case,
#         transformers=preprocessors,
#     )

#     logger.info(f"Getting datasets...")
#     ds_train, _, _ = get_image_dataset(get_image_dataset_params)()
#     logger.info(f"Done")

#     sample_frame, _ = ds_train[0]
#     sample_frames.append(sample_frame)

# import matplotlib.ticker as tck

# from skimage.metrics import normalized_mutual_information as nmi

# factors = range(1, 10)

# metrics = [
#     retained_variance,
#     shannon_cross_entropy_ratio,
#     shannon_entropy_ratio,
#     structural_similarity_ratio,
#     normalized_mutual_information,
#     nmi
# ]

# sns.set_style("whitegrid")

# f, axes = plt.subplots(len(metrics), len(sample_frames), figsize=(16, 16), sharex="row", sharey="col")

# x = factors
# preferred_factor = 4
# for col, sample_frame in enumerate(sample_frames):
#     downscaled_frames = [Downscaler(factor)(sample_frame) for factor in factors]

#     for row, metric in enumerate(metrics):
#         ax = axes[row, col]

#         y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

#         ax.scatter(x, y, s=20, color='k')
#         ax.scatter(x[0], y[0], facecolors="none", edgecolors="k", marker="$\odot$", s=100)
#         ax.scatter(x[preferred_factor], y[preferred_factor], facecolors="none", edgecolors="k", marker="$\odot$", s=100)

#         if not row:
#             ax.set_title(f"Dataset {col}")
#         if not col:
#             ax.set_ylabel(" ".join(metric.__name__.split("_")).title())

#         ax.xaxis.grid(True, which='minor')

"""### Consecutive frames"""

# TODO: fix this!!!
# since the dataset is already shuffled, I'm not taking consecutive frames

# TODO: test defining the hold-out sets as literal slices, as in: ds_train, ds_val, ds_test = ds[:X], ds[X:Y], ds[Y:]
# where ds is NOT shuffled


# frames_indices = tuple(range(10)) + tuple(range(10, 100, 10))

# metrics = [
#     retained_variance,
#     shannon_cross_entropy_ratio,
#     shannon_entropy_ratio,
#     structural_similarity_ratio,
#     normalized_mutual_information
# ]

# sns.set_style("whitegrid")

# f, axes = plt.subplots(len(metrics), len(boiling_direct_datasets), figsize=(16, 16), sharex="row", sharey="col")

# x = [index + 1 for index in frames_indices]
# for col, splits in enumerate(boiling_direct_datasets):
#     ds_train, _, _ = splits()
#     frames = features(ds_train).fetch(frames_indices)
#     for row, metric in enumerate(metrics):
#         ax = axes[row, col]

#         y = [metric(frames[0], frame) for frame in frames]

#         ax.scatter(x, y, s=20, color='k')
#         ax.scatter(x[0], y[0], facecolors="none", edgecolors="k", marker="$\odot$", s=100)

#         if not row:
#             ax.set_title(f"Dataset {col}")
#         if not col:
#             ax.set_ylabel(" ".join(metric.__name__.split("_")).title())

#         ax.xaxis.grid(True, which='minor')
#         ax.set_xscale("log")


# f, axes = plt.subplots(len(boiling_direct_datasets), 3, figsize=(10, 16), sharex="row", sharey="col")

# x = [index + 1 for index in frames_indices]
# for row, splits in enumerate(boiling_direct_datasets):
#     for col, split_name, split in zip(range(3), ("Train", "Val", "Test"), splits()):
#         ax = axes[row, col]
#         frame, data = split.shuffle()[0]

#         if not row:
#             ax.set_title(split_name)
#         if not col:
#             ax.set_ylabel(f"Dataset {row + 1}")

#         ax.set_xlabel(f"{data['Flux [W/cm**2]']:.2f}W/cm² (#{data['index']})")
#         ax.imshow(frame.squeeze(), cmap="gray")
#         ax.grid(False)

"""FIRST CONDENSATION CASE JUST FOR FUN"""

first_frame = condensation_dataset()[0][0]
with strategy_scope(strategy):
    architecture = hoboldnet2(first_frame.shape, dropout=0.5, normalize_images=True)

compiled_model = compile_model(
    architecture,
    get_baseline_compile_params(),
)
trained_model = fit_condensation_model(
    compiled_model,
    condensation_dataset,
    get_baseline_fit_params(),
    target='mass_rate',
)

"""### On-Wire Pool Boiling

#### Validation
"""

validated_model_direct = get_pretrained_baseline_boiling_model(direct=True, normalize_images=False)
print(validated_model_direct)
print('Evaluation:', validated_model_direct.evaluation)

validated_model_indirect = get_pretrained_baseline_boiling_model(
    direct=False, normalize_images=False
)
print(validated_model_indirect)
print('Evaluation:', validated_model_indirect.evaluation)

validated_model_direct_normalized = get_pretrained_baseline_boiling_model(
    direct=True, normalize_images=True
)
print(validated_model_direct_normalized)
print('Evaluation:', validated_model_direct_normalized.evaluation)

validated_model_indirect_normalized = get_pretrained_baseline_boiling_model(
    direct=False, normalize_images=True
)
print(validated_model_indirect_normalized)
print('Evaluation:', validated_model_indirect_normalized.evaluation)

# !pip install shap

# import shap

# import numpy as np

# background = np.array(features(baseline_boiling_dataset_direct()[0].sample(100)))
# samples = np.array(features(baseline_boiling_dataset_direct()[1].sample(4)))

# e = shap.DeepExplainer(validated_model_direct.architecture.model, background)

# shap_values = e.shap_values(samples, check_additivity=False)

# shap.image_plot(shap_values, samples)

# print(baseline_boiling_dataset_direct()[0][0][0].shape)
# print(baseline_boiling_dataset_indirect()[0][0][0].shape)

# import numpy as np

# background = np.array(features(baseline_boiling_dataset_indirect()[0].sample(100)))
# samples = np.array(features(baseline_boiling_dataset_indirect()[1].sample(4)))

# e = shap.DeepExplainer(validated_model_indirect.architecture.model, background)

# shap_values = e.shap_values(samples, check_additivity=False)

# shap.image_plot(shap_values, samples)

# # explain how the input to the 7th layer of the model explains the top two classes
# def map2layer(x, layer):
#     feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
#     return K.get_session().run(model.layers[layer].input, feed_dict)

# e = shap.GradientExplainer(
#     (model.layers[7].input, model.layers[-1].output),
#     map2layer(X, 7),
#     local_smoothing=0 # std dev of smoothing noise
# )
# shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)

# # get the names for the classes
# index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# # plot the explanations
# shap.image_plot(shap_values, to_explain, index_names)

# !pip install tf-explain

# from tf_explain.core.grad_cam import GradCAM

# explainer = GradCAM()
# grid = explainer.explain(baseline_boiling_dataset_direct()[1][0], model.architecture.model, )

"""#### Retrain randomness"""

# %%time
# %tensorboard --logdir $tensorboard_logs_path


logger.info('Analyzing effects of random initialization')

NUMBER_OF_RETRAINS = 7
evaluations = []
for retrain_index in range(NUMBER_OF_RETRAINS):
    logger.info('Compiling...')
    with strategy_scope(strategy):
        compiled_model = compile_model(
            baseline_boiling_model_architecture_direct, get_baseline_compile_params()
        )
    logger.info('Done')

    logger.info('Training...')
    fit_model_params = FitModelParams(
        batch_size=200,
        epochs=100,
        callbacks=LazyDescribed.from_list(
            [
                LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
                LazyDescribed.from_constructor(
                    tf.keras.callbacks.EarlyStopping,
                    monitor='val_loss',
                    min_delta=0,
                    patience=10,  # got error ~9 with Adam 1e-4
                    baseline=None,
                    mode='auto',
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
        ),
    )

    with strategy_scope(strategy):
        model = fit_boiling_model(
            compiled_model,
            baseline_boiling_dataset_direct,
            fit_model_params,
            target='Flux [W/cm**2]',
            try_id=retrain_index,
        )
    print(model)
    best_performance = min(model.history, key=lambda data: data['val_loss'])
    evaluations.append(best_performance)
    logger.info('Done')

pprint(evaluations)

"""#### Other wire - training"""

# %%time
# %tensorboard --logdir $tensorboard_logs_path


logger.info('Testing with other wire')

DATASET = boiling_direct_datasets[2]

logger.info('Compiling...')
first_frame, _ = DATASET()[0][0]

with strategy_scope(strategy):
    # TODO: replace this with utility function!
    architecture = hoboldnet2(first_frame.shape, dropout=0.5, normalize_images=False)
    compiled_model = compile_model(architecture, get_baseline_compile_params())
logger.info('Done')

logger.info('Training...')
fit_model_params = FitModelParams(
    batch_size=200,
    epochs=100,
    callbacks=LazyDescribed.from_list(
        [
            LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
            LazyDescribed.from_constructor(
                tf.keras.callbacks.EarlyStopping,
                monitor='val_loss',
                min_delta=0,
                patience=10,
                baseline=None,
                mode='auto',
                restore_best_weights=True,
                verbose=1,
            ),
        ]
    ),
)

with strategy_scope(strategy):
    model = fit_boiling_model(compiled_model, DATASET, fit_model_params, target='Flux [W/cm**2]')
pprint(model)
logger.info('Done')

"""#### Boiling learning curve"""

# %%time
# %tensorboard --logdir $tensorboard_logs_path


logger.info('Analyzing learning curve')

BATCH_SIZE = 200

ds_evaluation_direct = to_tensorflow(
    baseline_boiling_dataset_direct | subset('val'),
    batch_size=BATCH_SIZE,
    target='Flux [W/cm**2]',
)

ds_evaluation_indirect = to_tensorflow(
    baseline_boiling_dataset_indirect | subset('val'),
    batch_size=BATCH_SIZE,
    target='Flux [W/cm**2]',
)


@cache(JSONTableAllocator(analyses_path / 'studies' / 'boiling-learning-curve'))
def boiling_learning_curve_point(
    fraction: Fraction, *, direct: bool = True, normalize_images: bool = False
) -> dict[str, float]:
    logger.info(f'Analyzing fraction {fraction}')

    logger.info('Getting datasets...')
    datasets = (
        baseline_boiling_dataset_direct if direct else baseline_boiling_dataset_indirect
    ) | dataset_sampler(fraction)
    logger.info('Done')

    logger.info('Compiling...')
    with strategy_scope(strategy):
        compiled_model = compile_model(
            get_baseline_boiling_model(direct=direct, normalize_images=normalize_images),
            get_baseline_compile_params(),
        )
    logger.info('Done')

    logger.info('Training...')

    model = fit_boiling_model(
        compiled_model, datasets, get_baseline_fit_params(), target='Flux [W/cm**2]'
    )

    logger.info(f'Evaluating: {fraction}')
    with strategy_scope(strategy):
        compile_model(model.architecture, get_baseline_compile_params())
    evaluation = model.architecture.evaluate(
        (ds_evaluation_direct if direct else ds_evaluation_indirect)()
    )

    logger.info('Done')

    return evaluation


boiling_learning_curve = {
    fraction: boiling_learning_curve_point(
        fraction, direct=direct, normalize_images=normalize_images
    )
    for fraction in (
        Fraction(1, 100),
        Fraction(1, 20),
        Fraction(1, 10),
        Fraction(1, 2),
        Fraction(1, 1),
    )
    for direct in (False, True)
    for normalize_images in (False, True)
}


METRIC_NAMES = ('MSE', 'RMS', 'MAE', 'MAPE', 'R2')

console = Console()

learning_curve_analysis = Table(title='Learning curve analysis')
learning_curve_analysis.add_column('Dataset fraction', justify='center')

for metric_name in METRIC_NAMES:
    learning_curve_analysis.add_column(metric_name, justify='right')

for fraction, metrics in boiling_learning_curve.items():
    learning_curve_analysis.add_row(
        str(fraction), *(f'{metrics[metric_name]:.2f}' for metric_name in METRIC_NAMES)
    )

console.print(learning_curve_analysis)

"""#### Auto ML"""

regular_wire_best_model_direct_visualization = autofit_to_dataset(
    baseline_boiling_dataset_direct,
    target=boiling_target_name,
    normalize_images=True,
    max_model_size=baseline_boiling_model_direct_size,
    goal=None,
)

print(regular_wire_best_model_direct_visualization)

regular_wire_best_model_indirect_visualization = autofit_to_dataset(
    baseline_boiling_dataset_indirect,
    target=boiling_target_name,
    normalize_images=True,
    max_model_size=baseline_boiling_model_indirect_size,
    goal=None,
)

print(regular_wire_best_model_indirect_visualization)

"""#### AutoML - Less data"""


ds_train = baseline_boiling_dataset_direct | dataset_sampler(Fraction(1, 100)) | subset('train')
ds_val = baseline_boiling_dataset_direct | subset('val')

regular_wire_best_model_direct_visualization_less_data = autofit_to_dataset(
    LazyDescribed.from_value_and_description(
        (ds_train(), ds_val(), None), (ds_train, ds_val, None)
    ),
    target=boiling_target_name,
    normalize_images=True,
    max_model_size=baseline_boiling_model_indirect_size,
    goal=None,
)

print(regular_wire_best_model_direct_visualization_less_data)

"""#### Other wire - auto ML"""


# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# hypermodel = ConvImageRegressor(
#     loss=loss,
#     metrics=metrics,
#     tuner=EarlyStoppingGreedy,
#     directory=hypermodel_allocator,
#     max_model_size=int(
#         baseline_boiling_model_architecture.count_parameters(trainable=True, non_trainable=False)
#     ),
#     strategy=strategy,
#     goal=baseline_boiling_loss,
#     normalize_images=False,
# )

# tune_model_params = TuneModelParams(
#     batch_size=16,
#     callbacks=Described.from_list(
#         [
#             Described.from_constructor(tf.keras.callbacks.TerminateOnNaN, P()),
#             Described.from_constructor(
#                 tf.keras.callbacks.EarlyStopping,
#                 P(
#                     monitor='val_loss',
#                     min_delta=0,
#                     # patience=2,
#                     patience=10,
#                     baseline=None,
#                     mode='auto',
#                     restore_best_weights=True,
#                     verbose=1,
#                 ),
#             ),
#         ]
#     ),
# )

# regular_wire_best_model = autofit(
#     hypermodel,
#     datasets=boiling_direct_datasets[1],
#     params=tune_model_params,
#     target='Flux [W/cm**2]',
# )
# print(regular_wire_best_model)


logger.info('Analyzing cross-surface boiling evaluation')
BATCH_SIZE = 200


@cache(JSONTableAllocator(analyses_path / 'studies' / 'boiling-cross-surface'))
def boiling_cross_surface_evaluation(
    direct_visualization: bool,
    training_cases: tuple[int, ...],
    evaluation_cases: tuple[int, ...],
    normalize_images: bool = True,
) -> dict[str, float]:
    logger.info(
        f'Training on cases {training_cases} '
        f'| evaluation on {evaluation_cases} '
        f"| {'Direct' if direct_visualization else 'Indirect'} visualization"
    )

    all_datasets = boiling_direct_datasets if direct_visualization else boiling_indirect_datasets
    training_datasets = tuple(all_datasets[training_case] for training_case in training_cases)
    evaluation_datasets = tuple(
        all_datasets[evaluation_case] for evaluation_case in evaluation_cases
    )

    training_dataset = LazyDescribed.from_describable(training_datasets) | datasets_merger()
    evaluation_dataset = LazyDescribed.from_describable(evaluation_datasets) | datasets_merger()

    with strategy_scope(strategy):
        architecture = get_baseline_boiling_model(
            direct=direct_visualization,
            normalize_images=normalize_images,
        )
        compiled_model = compile_model(architecture, get_baseline_compile_params())

    logger.info('Training...')

    model = fit_boiling_model(
        compiled_model, training_dataset, get_baseline_fit_params(), target='Flux [W/cm**2]'
    )

    logger.info('Evaluating')
    with strategy_scope(strategy):
        compile_model(model.architecture, get_baseline_compile_params())

    ds_evaluation_val = to_tensorflow(
        evaluation_dataset | subset('val'),
        batch_size=BATCH_SIZE,
        target='Flux [W/cm**2]',
    )

    evaluation = model.architecture.evaluate(ds_evaluation_val())
    logger.info(f'Done: {evaluation}')

    return evaluation


cases_indices = ((0,), (1,), (0, 1), (2,), (3,), (2, 3), (0, 1, 2, 3))

boiling_cross_surface = {
    (is_direct, training_cases, evaluation_cases): boiling_cross_surface_evaluation(
        is_direct, training_cases, evaluation_cases, normalize_images=True
    )
    for is_direct, training_cases, evaluation_cases in itertools.product(
        (False, True), cases_indices, cases_indices
    )
}

print(boiling_cross_surface)


console = Console()


def _format_sets(indices: tuple[int, ...]) -> str:
    return ' + '.join(map(str, indices))


def _get_and_format_results(direct_result: float, indirect_result: float) -> str:
    formatted_direct_result = f'[bold]{direct_result:.4f}[/bold]'
    formatted_indirect_result = f'{indirect_result:.4f}'

    ratio = (indirect_result - direct_result) / direct_result
    formatted_ratio = (
        f'[bold][bright_red]{ratio:+.2%}[/bright_red][/bold]'
        if ratio > 0
        else f'[bold][bright_green]{ratio:+.2%}[/bright_green][/bold]'
    )

    return f'{formatted_direct_result}\n{formatted_indirect_result}\n({formatted_ratio})'


for metric_name in ('MSE', 'MAPE', 'RMS', 'R2'):
    cross_surface_analysis = Table(
        'Train \\ Eval',
        *(map(_format_sets, cases_indices)),
        title=f'Cross surface analysis - {metric_name}',
    )

    for training_indices in cases_indices:
        cross_surface_analysis.add_row(
            _format_sets(training_indices),
            *map(
                _get_and_format_results,
                (
                    boiling_cross_surface[(True, training_indices, evaluation_cases)][metric_name]
                    for evaluation_cases in cases_indices
                ),
                (
                    boiling_cross_surface[(False, training_indices, evaluation_cases)][metric_name]
                    for evaluation_cases in cases_indices
                ),
            ),
            end_section=True,
        )

    console.print(cross_surface_analysis)

"""#### Cross-surface boiling evaluation with AutoML"""

# %tensorboard --logdir $tensorboard_logs_path


# logger.info('Analyzing cross-surface boiling evaluation with AutoML')
# BATCH_SIZE = 200


# @cache(JSONTableAllocator(analyses_path / 'studies' / 'boiling-cross-surface-automl'))
# def boiling_cross_surface_evaluation_automl(
#     direct_visualization: bool, training_cases: tuple[int, ...], evaluation_cases: tuple[int, ...]
# ) -> dict[str, float]:
#     logger.info(
#         f'Training on cases {training_cases} '
#         f'| evaluation on {evaluation_cases} '
#         f"| {'Direct' if direct_visualization else 'Indirect'} visualization"
#     )

#     all_datasets = boiling_direct_datasets if direct_visualization else boiling_indirect_datasets
#     training_datasets = tuple(all_datasets[training_case] for training_case in training_cases)
#     evaluation_datasets = tuple(
#         all_datasets[evaluation_case] for evaluation_case in evaluation_cases
#     )

#     training_dataset = LazyDescribed.from_describable(training_datasets) | datasets_merger()
#     evaluation_dataset = LazyDescribed.from_describable(evaluation_datasets) | datasets_merger()

#     tune_model_return: TuneModelReturn = autofit_to_dataset(
#         training_dataset,
#         target=boiling_target_name,
#         normalize_images=True,
#         max_model_size=baseline_boiling_model_direct_size,
#         goal=None,
#     )

#     logger.info('Evaluating')
#     with strategy_scope(strategy):
#         compile_model(tune_model_return.model, get_baseline_compile_params())

#     ds_evaluation_val = to_tensorflow(
#         evaluation_dataset | subset('val'),
#         batch_size=BATCH_SIZE,
#         target='Flux [W/cm**2]',
#     )

#     evaluation = tune_model_return.model.evaluate(ds_evaluation_val())
#     logger.info(f'Done: {evaluation}')

#     return evaluation


# cases_indices = ((0,), (1,), (0, 1), (2,), (3,), (2, 3), (0, 1, 2, 3))

# boiling_cross_surface_automl = {
#     (is_direct, training_cases, evaluation_cases): boiling_cross_surface_evaluation_automl(
#         is_direct, training_cases, evaluation_cases
#     )
#     for is_direct, training_cases, evaluation_cases in itertools.product(
#         (False, True), cases_indices, cases_indices
#     )
# }

condensation_all_cases = ExperimentVideoDataset.make_union(
    *set_condensation_datasets_data.main(
        condensation_datasets,
        condensation_data_spec_path,
    )
)

get_image_dataset_params = GetImageDatasetParams(
    condensation_all_cases,
    transformers=condensation_preprocessors,
    # dataset_size=Fraction(1, 100),
    dataset_size=None,
)

ds_train, ds_val, ds_test = get_image_dataset(get_image_dataset_params)

"""#### First Condensation Case"""

# TODO: interesting analysis:
# since the parametric studies use the same type of surface, I would expect that the network would get more confused

# REGRESSION


logger.info('First condensation case')


# condensation_all_cases = ExperimentVideoDataset.make_union(
#     *set_condensation_datasets_data.main(
#         condensation_datasets,
#         condensation_data_spec_path,
#     )
# )

condensation_all_cases = ExperimentVideoDataset.make_union(*condensation_datasets)

get_image_dataset_params = GetImageDatasetParams(
    condensation_all_cases,
    transformers=condensation_preprocessors,
    dataset_size=None,
)

logger.info('Getting datasets...')
# TODO: this should be set by `set_condensation_datasets_data`
def _set_case_name(data: dict[str, Any]) -> dict[str, Any]:
    data['case_name'] = ':'.join(data['name'].split(':')[:2])
    return data


ds_train, ds_val, ds_test = get_image_dataset(get_image_dataset_params)
ds_train = map_targets(ds_train, _set_case_name)
ds_val = map_targets(ds_val, _set_case_name)
ds_test = map_targets(ds_test, _set_case_name)
logger.info('Done')

logger.info('Calculating classes')
CLASSES = sorted(frozenset(targets(ds_train).map(itemgetter('case_name')).prefetch(4096)))
N_CLASSES = len(CLASSES)


def _set_case(data: dict[str, Any]) -> dict[str, Any]:
    data['case'] = CLASSES.index(data['case_name'])
    return data


ds_train = map_targets(ds_train, _set_case)
ds_val = map_targets(ds_val, _set_case)
ds_test = map_targets(ds_test, _set_case)
logger.info('Done')

logger.info('Describing datasets...')
datasets = Described(value=(ds_train, ds_val, ds_test), description=get_image_dataset_params)
logger.info('Done')

logger.info('Getting first frame...')
first_frame, _ = ds_train[0]
logger.info('Done')

logger.info('Compiling...')
with strategy_scope(strategy):
    architecture = hoboldnet2(
        first_frame.shape,
        dropout=0.5,
        output_layer_policy='float32',
        problem=ProblemType.CLASSIFICATION,
        num_classes=N_CLASSES,
    )
    compile_params = CompileModelParams(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name='top3'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5'),
            # tfa.metrics.F1Score(N_CLASSES, name='F1'),
        ],
    )
    compiled_model = compile_model(architecture, compile_params)
logger.info('Done')

logger.info('Training...')
fit_model_params = FitModelParams(
    batch_size=200,
    epochs=100,
    callbacks=Described.from_list(
        [
            Described.from_constructor(tf.keras.callbacks.TerminateOnNaN, P()),
            Described.from_constructor(
                tf.keras.callbacks.EarlyStopping,
                P(
                    monitor='val_loss',
                    min_delta=0,
                    patience=2,
                    # patience=10,
                    baseline=None,
                    mode='auto',
                    restore_best_weights=True,
                    verbose=1,
                ),
            ),
            # Described.from_constructor(
            #     ReduceLROnPlateau,
            #     P(
            #         monitor='val_loss',
            #         factor=0.5,
            #         patience=2,
            #         min_delta=0.01,
            #         min_delta_mode='relative',
            #         min_lr=0,
            #         mode='auto',
            #         cooldown=2,
            #     ),
            # ),
        ]
    ),
)

with strategy_scope(strategy):
    model = fit_condensation_model(compiled_model, datasets, fit_model_params)
print(model)
logger.info('Done')

# REGRESSION


logger.info('First condensation case')

# condensation_all_cases = ExperimentVideoDataset.make_union(
#     *set_condensation_datasets_data.main(
#         condensation_datasets,
#         condensation_data_spec_path,
#     )
# )

get_image_dataset_params = GetImageDatasetParams(
    condensation_all_cases,
    transformers=condensation_preprocessors,
    dataset_size=None,
)

logger.info('Getting datasets...')
ds_train, ds_val, ds_test = get_image_dataset(get_image_dataset_params)
logger.info('Done')

logger.info('Describing datasets...')
datasets = Described(value=(ds_train, ds_val, ds_test), description=get_image_dataset_params)
logger.info('Done')

logger.info('Getting first frame...')
first_frame, _ = ds_train[0]
logger.info('Done')

logger.info('Compiling...')
with strategy_scope(strategy):
    architecture = hoboldnet2(
        first_frame.shape,
        dropout=0.5,
        output_layer_policy='float32',
    )
    compiled_model = compile_model(architecture, get_baseline_compile_params())
logger.info('Done')

logger.info('Training...')
fit_model_params = FitModelParams(
    batch_size=200,
    epochs=100,
    callbacks=Described.from_list(
        [
            Described.from_constructor(tf.keras.callbacks.TerminateOnNaN, P()),
            Described.from_constructor(
                tf.keras.callbacks.EarlyStopping,
                P(
                    monitor='val_loss',
                    min_delta=0,
                    patience=2,
                    # patience=10,
                    baseline=None,
                    mode='auto',
                    restore_best_weights=True,
                    verbose=1,
                ),
            ),
            # Described.from_constructor(
            #     ReduceLROnPlateau,
            #     P(
            #         monitor='val_loss',
            #         factor=0.5,
            #         patience=2,
            #         min_delta=0.01,
            #         min_delta_mode='relative',
            #         min_lr=0,
            #         mode='auto',
            #         cooldown=2,
            #     ),
            # ),
        ]
    ),
)

with strategy_scope(strategy):
    model = fit_condensation_model(compiled_model, datasets, fit_model_params, target='mass_rate')
print(model)
logger.info('Done')

"""## Studies

### Boiling learning curve
"""

# TODO: move it here!!!

assert False, 'STOP!'

# """## The End

# ## Experimental Code

# ### Kramer data
# """


# """### AutoKeras"""


# BATCH_SIZE = 32

# get_image_dataset_params = GetImageDatasetParams(
#     boiling_cases_timed[0],
#     transformers=(*boiling_direct_preprocessors, ImageNormalizer()),
#     dataset_size=None,
# )

# logger.info('Getting datasets...')
# datasets = Described(get_image_dataset(get_image_dataset_params), get_image_dataset_params)
# ds_train, ds_val, ds_test = datasets.value
# first_frame, _ = ds_train[0]
# ds_train, ds_val, _ = to_tensorflow_triplet(
#     datasets, batch_size=BATCH_SIZE, include_test=False, target='Flux [W/cm**2]'
# )
# ds_train = ds_train.unbatch().prefetch(tf.data.AUTOTUNE)
# ds_val = ds_val.unbatch().prefetch(tf.data.AUTOTUNE)
# logger.info('Done')

# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# inputs = ak.ImageInput()
# x = LayersBlock(baseline_boiling_model_architecture.model.layers[1:-3])(inputs)
# outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(x)

# regressor = ak.AutoModel(
#     inputs,
#     outputs,
#     distribution_strategy=strategy.value,
#     tuner=EarlyStoppingGreedy,
#     overwrite=True,
#     goal=10,
#     # max_model_size=sum(
#     #     tf.keras.backend.count_params(p)
#     #     for p in baseline_boiling_model_architecture.model.trainable_weights
#     # )
# )


# regressor.fit(
#     ds_train,
#     validation_data=ds_val,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.TerminateOnNaN(),
#         # TimePrinter()
#     ],
#     batch_size=BATCH_SIZE,
# )


# logger.info('Done')


# BATCH_SIZE = 32

# get_image_dataset_params = GetImageDatasetParams(
#     boiling_cases_timed[0],
#     transformers=(*boiling_direct_preprocessors, ImageNormalizer()),
#     dataset_size=None,
# )

# logger.info('Getting datasets...')
# datasets = Described(get_image_dataset(get_image_dataset_params), get_image_dataset_params)
# ds_train, ds_val, ds_test = datasets.value
# first_frame, _ = ds_train[0]
# ds_train, ds_val, _ = to_tensorflow_triplet(
#     datasets, batch_size=BATCH_SIZE, include_test=False, target='Flux [W/cm**2]'
# )
# ds_train = ds_train.unbatch().prefetch(tf.data.AUTOTUNE)
# ds_val = ds_val.unbatch().prefetch(tf.data.AUTOTUNE)
# logger.info('Done')

# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# inputs = ak.ImageInput()
# x = ak.ImageBlock(normalize=False, augment=False)(inputs)
# x = ak.SpatialReduction()(x)
# x = ak.DenseBlock()(x)
# outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(x)

# regressor = ak.AutoModel(
#     inputs, outputs, distribution_strategy=strategy.value, tuner=BetterGreedy, overwrite=True
# )

# regressor.fit(
#     ds_train,
#     validation_data=ds_val,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.TerminateOnNaN(),
#         # TimePrinter()
#     ],
#     batch_size=BATCH_SIZE,
# )


# logger.info('Done')


# save_best_epoch_on_epoch_end = SaveBestEpoch.on_epoch_end
# SaveBestEpoch.on_epoch_end = Callback.on_epoch_end


# class CNN2Block(ak.Block):
#     def build(self, hp, inputs=None):
#         # Get the input_node from inputs.
#         input_node = tf.nest.flatten(inputs)[0]
#         x = Conv2D(32, (5, 5), padding='same')(input_node)
#         x = ReLU()(x)
#         x = MaxPool2D((2, 2), strides=(2, 2))(x)
#         x = Dropout(0.5)(x)
#         # x = Dropout(hp.Float("dropout", min_value=0.0, max_value=1.0))(x)
#         return x


# class DenseBlock(ak.Block):
#     def build(self, hp, inputs=None):
#         # Get the input_node from inputs.
#         input_node = tf.nest.flatten(inputs)[0]
#         x = Dense(200)(input_node)
#         x = ReLU()(x)
#         x = Dropout(0.5)(x)
#         # x = Dropout(hp.Float("dropout", min_value=0.0, max_value=1.0))(x)
#         x = Dense(1)(x)
#         return Activation('linear')(x)


# class HoboldNetBlock(ak.Block):
#     def build(self, hp, inputs=None):
#         # Get the input_node from inputs.
#         input_node = tf.nest.flatten(inputs)[0]
#         x = Conv2D(32, (5, 5), padding='same', dtype='mixed_float16')(input_node)
#         x = ReLU(dtype='mixed_float16')(x)
#         x = MaxPool2D((2, 2), strides=(2, 2), dtype='mixed_float16')(x)
#         x = Flatten(dtype='mixed_float16')(x)
#         x = Dropout(0.5, dtype='mixed_float16')(x)
#         x = Dense(200, dtype='mixed_float16')(x)
#         x = ReLU(dtype='mixed_float16')(x)
#         return x
#         # x = Dropout(0.5)(x)
#         # # x = Dropout(hp.Float("dropout", min_value=0.0, max_value=1.0))(x)
#         # x = Dense(1)(x)
#         # return Activation('linear')(x)


# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# input_node = ak.ImageInput()
# # x = ak.Normalization()(input_node)
# # x = CNN2Block()(input_node)
# # x = ak.SpatialReduction()(x)
# # x = DenseBlock()(x)
# # output_node = ak.RegressionHead()(x)
# # x = HoboldNetBlock()(input_node)
# # output_node = DenseBlock()(x)
# x = HoboldNetBlock()(input_node)
# output_node = ak.RegressionHead(loss=loss, dropout=0.5, output_dim=1, metrics=metrics)(x)

# auto_model = ak.AutoModel(
#     inputs=input_node,
#     outputs=output_node,
#     directory=analyses_path / 'temp' / 'auto_tune_hoboldnet2-5',
#     overwrite=True,
#     distribution_strategy=strategy.value,
# )
# auto_model.fit(
#     ds_train.batch(
#         200,
#         num_parallel_calls=tf.data.AUTOTUNE,
#         deterministic=False,
#     ).prefetch(tf.data.AUTOTUNE),
#     validation_data=ds_val.batch(
#         200,
#         num_parallel_calls=tf.data.AUTOTUNE,
#         deterministic=False,
#     ).prefetch(tf.data.AUTOTUNE),
#     # batch_size=200,
#     callbacks=[
#         tf.keras.callbacks.TerminateOnNaN(),
#         TimePrinter(
#             when={
#                 # 'on_batch_begin',
#                 # 'on_batch_end',
#                 'on_epoch_begin',
#                 'on_epoch_end',
#                 # 'on_predict_batch_begin',
#                 # 'on_predict_batch_end',
#                 # 'on_predict_begin',
#                 # 'on_predict_end',
#                 # 'on_test_batch_begin',
#                 # 'on_test_batch_end',
#                 # 'on_test_begin',
#                 # 'on_test_end',
#                 # 'on_train_batch_begin',
#                 # 'on_train_batch_end',
#                 # 'on_train_begin',
#                 # 'on_train_end',
#             }
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             min_delta=0,
#             patience=2,
#             # patience=10,
#             baseline=None,
#             mode='auto',
#             restore_best_weights=True,
#             verbose=1,
#         ),
#     ],
# )

# """## Old Code"""

# reference_datasets_boiling = (
#     boiling_cases_timed[0]['GOPR2868'].as_tf_dataset(),
#     boiling_cases_timed[1]['GOPR2878'].as_tf_dataset(),
#     boiling_cases_timed[2]['GOPR2908'].as_tf_dataset(),
#     boiling_cases_timed[3]['GOPR2948'].as_tf_dataset(),
# )
# reference_datasets_condensation = (
#     condensation_datasets_dict['stainless steel:polished'][
#         'stainless steel:polished:test 4:00003'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:old']['parametric:old:test 3:00006'].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_inf 40C'][
#         'parametric:T_inf 40C:test 2:00005'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_inf 60C'][
#         'parametric:T_inf 60C:test 1:00001'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_s 5C'][
#         'parametric:T_s 5C:test 3:00008'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_s 20C'][
#         'parametric:T_s 20C:test 3:00008'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:rh 70%'][
#         'parametric:rh 70%:test 3:00001'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:rh 90%'][
#         'parametric:rh 90%:test 3:00009'
#     ].as_tf_dataset(),
# )

# """### FPS"""

# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline


# # TODO: is FPS 30 or 1 here???
# # Answer: take a look at <https://drive.google.com/drive/u/1/folders/1hVLDeLOlklVqIN-W6eGbRUqTxMXZTF74>
# # It seems that FPS is already 1, so frame #30 happens 30s after frame #0.

# if OPTIONS['analyze_consecutive_frames']:
#     for reference_datasets, preprocessors, final_timeshift, timeshifts in (
#         (
#             reference_datasets_boiling,
#             boiling_direct_preprocessors,
#             1,
#             (0, 1, 2, 3, 5, 10, 20, 30, 60),
#         ),
#         (
#             reference_datasets_condensation,
#             condensation_preprocessors,
#             300,
#             (0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1600, 3600, 7200),
#         ),
#     ):
#         for dataset in reference_datasets:
#             n_frames = max(timeshifts) + 1

#             preprocessors = select_preprocessors(preprocessors)

#             dataset = bl.datasets.apply_transformers(dataset, preprocessors)
#             _, data = list(dataset.take(1).as_numpy_iterator())[0]
#             print(data)
#             frames = {
#                 idx: frame
#                 for idx, frame in enumerate(
#                     dataset.map(lambda image, data: image).take(n_frames).as_numpy_iterator()
#                 )
#                 if idx in timeshifts
#             }

#             fig = plt.figure()
#             ax = fig.add_subplot(1, 1, 1)
#             ax.imshow(frames[0], cmap='gray')
#             fig.show()

#             analyze_consecutive_frames.main(
#                 frames.items(),
#                 metrics={
#                     'Retained variance': retained_variance,
#                     'Cross-entropy ratio': shannon_cross_entropy_ratio,
#                     'Entropy ratio': shannon_entropy_ratio,
#                     'NMI ratio': normalized_mutual_information,
#                     'Structural similarity': structural_similarity_ratio,
#                 },
#                 timeshifts=timeshifts,
#                 final_timeshift=final_timeshift,
#                 xscale='symlog',
#                 figsize=(4, 3),
#             )

# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Source: https://stackoverflow.com/questions/10917495/matplotlib-imshow-in-3d-plot

# # plt.clf()
# # fig = plt.figure(1)
# # ax = fig.gca(projection='3d')

# # params_no_reduce_lr = []
# # metrics_no_reduce_lr = []
# # params_reduce_lr = []
# # metrics_reduce_lr = []

# # for params, metrics in evaluations:
# #     if params['reduce_lr_on_plateau']:
# #         params_reduce_lr.append(params)
# #         metrics_reduce_lr.append(metrics)
# #     else:
# #         params_no_reduce_lr.append(params)
# #         metrics_no_reduce_lr.append(metrics)

# # X_label = 'batch_size'
# # Y_label = 'lr'
# # Z_label = 'MAE'

# # for reduce_lr in (False, True):
# #     params = {
# #         False: params_no_reduce_lr,
# #         True: params_reduce_lr
# #     }[reduce_lr]

# #     metrics = {
# #         False: metrics_no_reduce_lr,
# #         True: metrics_reduce_lr
# #     }[reduce_lr]

# #     X_list = list(set(funcy.pluck(X_label, params)))
# #     Y_list = list(set(funcy.pluck(Y_label, params)))

# #     X = np.zeros((len(X_list),))
# #     Y = np.zeros((len(Y_list),))
# #     Z = np.zeros((X.shape[0], Y.shape[0]))

# #     for (i, x), (j, y) in it.product(
# #             enumerate(X_list),
# #             enumerate(Y_list)
# #     ):
# #         X[i] = x
# #         Y[j] = y


# #         Z[i, j] =

# #     Z = np.array(
# #         funcy.pluck(
# #             Z_label,
# #             funcy.pluck(
# #                 'ds_val_gt10',
# #                 metrics
# #             )
# #         )
# #     )

# #     ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

# #     cset = ax.contourf(X, Y, Z, zdir='z', offset=min(Z),
# #             levels=np.linspace(min(Z),max(Z),100),cmap=plt.cm.jet)
# #     cset = ax.contourf(X, Y, Z, zdir='x', offset=min(X), cmap=plt.cm.jet)
# #     cset = ax.contourf(X, Y, Z, zdir='y', offset=max(Y), cmap=plt.cm.jet)

# #     ax.set_xlabel(X_label)
# #     ax.set_xlim(min(X), max(X))
# #     ax.set_ylabel(Y_label)
# #     ax.set_ylim(min(Y), max(Y))
# #     ax.set_zlabel(Z_label)
# #     ax.set_zlim(min(Z), max(Z))

# #     plt.show()

# """### Learning curve

# Evaluate models trained with different fractions of the dataset. For instance, 1%, 5%, 10%, 50% and 100% of the data.

# Tasks:

# - do some research: search for papers and books that explain this.

# ### Cross evaluation
# """

# metric_name = 'MAE'
# bar_metrics = dict(evaluations)
# bar_metrics = {
#     pack_: {set_name: set_metrics[metric_name] for set_name, set_metrics in sets.items()}
#     for pack_, sets in bar_metrics.items()
# }

# for k, v in bar_metrics.items():
#     print(k)
#     pprint(v)

# new_bar_metrics = {}
# for pack_, metrics_ in bar_metrics.items():
#     print('Pack:', pack_)
#     key = pack_.omit(['lr'])
#     print(key)
#     if key not in new_bar_metrics:
#         print('key not in new_bar_metrics')
#         new_bar_metrics[key] = {}
#     print(metrics_)
#     new_bar_metrics[key][pack_['lr']] = metrics_

# for pack_, metrics_dict in new_bar_metrics.items():
#     new_bar_metrics[pack_] = max(metrics_dict.values(), key=operator.itemgetter('ds_eval'))

# bar_metrics = {}
# for pack_, metrics_ in new_bar_metrics.items():
#     key = pack_.omit(['direct'])
#     print(key)
#     if key not in bar_metrics:
#         print('key not in bar_metrics')
#         bar_metrics[key] = {}
#     print(metrics_)
#     bar_metrics[key][pack_['direct']] = metrics_

# pprint(bar_metrics)

# # %matplotlib inline

# sns.set_context('notebook')
# sns.set_theme(style='ticks')

# metrics_data = pd.DataFrame.from_records(
#     [
#         {
#             'train': '+'.join(case_name[-1] for case_name in pack_['train_cases']),
#             'eval': '+'.join(case_name[-1] for case_name in pack_['eval_cases']),
#             'visualization': 'direct' if direct else 'indirect',
#             'metric': metrics_['ds_eval'],
#         }
#         for pack_, metrics_dict in bar_metrics.items()
#         for direct, metrics_ in metrics_dict.items()
#     ]
# )

# tips = sns.load_dataset('tips')

# # weird workaround, I don't know why metrics_data gives an error...
# new_metrics_data = tips.sample(len(metrics_data)).copy()
# new_metrics_data = new_metrics_data.rename(
#     columns={
#         'sex': 'train',
#         'total_bill': 'metric',
#         'smoker': 'eval',
#         'time': 'direct',
#     }
# )
# new_metrics_data['train'] = metrics_data['train']
# new_metrics_data['metric'] = metrics_data['metric']
# new_metrics_data['eval'] = metrics_data['eval']
# new_metrics_data['visualization'] = metrics_data['visualization']

# orient = 'v'
# x, y = ('train', 'metric')
# if orient == 'h':
#     x, y = y, x

# g = sns.catplot(
#     x=x,
#     y=y,
#     col='visualization',
#     data=new_metrics_data,
#     kind='bar',
#     hue='eval',
#     orient=orient,
# )
# label = 'MAE [W/cm²]'
# if orient == 'h':
#     g.set_xlabels(label)
# else:
#     g.set_ylabels(label)

# for ax in g.axes_dict.values():
#     for p, txt in zip(
#         ax.patches,
#         mit.flatten(
#             [
#                 itertools.repeat(txt, len(ax.patches) // len(g.legend.get_texts()))
#                 for txt in g.legend.get_texts()
#             ]
#         ),
#     ):
#         ax.text(
#             # p.get_x() - 0.01,
#             p.get_x(),
#             p.get_height() * 1.02,
#             txt.get_text(),
#             color='black',
#             rotation='horizontal',
#             size='large',
#         )


# def main(evaluations, metric_name: str):
#     bar_metrics = dict(evaluations)
#     bar_metrics = {
#         pack_: {set_name: set_metrics[metric_name] for set_name, set_metrics in sets.items()}
#         for pack_, sets in bar_metrics.items()
#     }

#     for k, v in bar_metrics.items():
#         print(k)
#         pprint(v)

#     new_bar_metrics = {}
#     for pack_, metrics_ in bar_metrics.items():
#         print('Pack:', pack_)
#         key = pack_.omit(['lr'])
#         print(key)
#         if key not in new_bar_metrics:
#             print('key not in new_bar_metrics')
#             new_bar_metrics[key] = {}
#         print(metrics_)
#         new_bar_metrics[key][pack_['lr']] = metrics_

#     for pack_, metrics_dict in new_bar_metrics.items():
#         new_bar_metrics[pack_] = max(metrics_dict.values(), key=operator.itemgetter('ds_eval'))

#     bar_metrics = {}
#     for pack_, metrics_ in new_bar_metrics.items():
#         key = pack_.omit(['direct'])
#         print(key)
#         if key not in bar_metrics:
#             print('key not in bar_metrics')
#             bar_metrics[key] = {}
#         print(metrics_)
#         bar_metrics[key][pack_['direct']] = metrics_

#     pprint(bar_metrics)

#     # %matplotlib inline

#     sns.set_context('notebook')
#     sns.set_theme(style='ticks')

#     metrics_data = pd.DataFrame.from_records(
#         [
#             {
#                 'train': '+'.join(case_name[-1] for case_name in pack_['train_cases']),
#                 'eval': '+'.join(case_name[-1] for case_name in pack_['eval_cases']),
#                 'visualization': 'direct' if direct else 'indirect',
#                 'metric': metrics_['ds_eval'],
#             }
#             for pack_, metrics_dict in bar_metrics.items()
#             for direct, metrics_ in metrics_dict.items()
#         ]
#     )

#     tips = sns.load_dataset('tips')

#     # weird workaround, I don't know why metrics_data gives an error...
#     new_metrics_data = tips.sample(len(metrics_data)).copy()
#     new_metrics_data = new_metrics_data.rename(
#         columns={
#             'sex': 'train',
#             'total_bill': 'metric',
#             'smoker': 'eval',
#             'time': 'direct',
#         }
#     )
#     new_metrics_data['train'] = metrics_data['train']
#     new_metrics_data['metric'] = metrics_data['metric']
#     new_metrics_data['eval'] = metrics_data['eval']
#     new_metrics_data['visualization'] = metrics_data['visualization']

#     orient = 'v'
#     x, y = ('train', 'metric')
#     if orient == 'h':
#         x, y = y, x
#     g = sns.catplot(
#         x=x,
#         y=y,
#         col='visualization',
#         data=new_metrics_data,
#         kind='bar',
#         hue='eval',
#         orient=orient,
#     )
#     label = 'MAE [W/cm²]'
#     if orient == 'h':
#         g.set_xlabels(label)
#     else:
#         g.set_ylabels(label)

#     for ax in g.axes_dict.values():
#         for p, txt in zip(
#             ax.patches,
#             mit.flatten(
#                 [
#                     itertools.repeat(txt, len(ax.patches) // len(g.legend.get_texts()))
#                     for txt in g.legend.get_texts()
#                 ]
#             ),
#         ):
#             ax.text(
#                 # p.get_x() - 0.01,
#                 p.get_x(),
#                 p.get_height() * 1.02,
#                 txt.get_text(),
#                 color='black',
#                 rotation='horizontal',
#                 size='large',
#             )


# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline

# sns.set_context('notebook')
# sns.set_theme(style='ticks')


# main(evaluations, 'MAE')

# """2+3 uses too much data!!!
# perhaps 2+3 could be defined as 50% of case 2 + 50% of case 3?
# """

# # # Fix the x-axes.
# # ax.set_xticks(x + bar_width)
# # ax.set_xticks(x)
# # ax.set_xticklabels(
# #     [
# #         '\n'.join([
# #             'train: ' + '+'.join(case_name[-1] for case_name in pack_['train_cases']),
# #             'eval: ' + '+'.join(case_name[-1] for case_name in pack_['eval_cases'])
# #         ])
# #         for pack_ in bar_metrics
# #     ],
# #     fontdict=dict(fontsize=10)
# # )

# # # Add legend.
# # ax.legend()

# # # Axis styling.
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_color('#DDDDDD')
# # ax.tick_params(bottom=False, left=False)
# # ax.set_axisbelow(True)
# # ax.yaxis.grid(True, color='#EEEEEE')
# # ax.xaxis.grid(False)

# # # Add axis and chart labels.
# # ax.set_xlabel('Job', labelpad=15)
# # ax.set_ylabel('MAE [W/cm²]', labelpad=15)
# # ax.set_title(f'Cross-evaluation metric', pad=15)

# # fig.tight_layout()
# # fig.show()

# """## TO-DO List

# - [ ] improve sectioning
# - [ ] remove old code
# - [ ] remove unnecessary comments
# - [ ] make "Boiling Learning.ipynb" deprecated
# - [ ] see available models at https://github.com/tensorflow/models/tree/master/official
# - [ ] see more models at https://tfhub.dev/tensorflow/efficientnet/b0/classification/1
# - [ ] see TensorFlow Addons, such as the following optimizer:
#     https://www.tensorflow.org/addons/tutorials/optimizers_lazyadam
# - [ ] include
#     https://www.tensorflow.org/addons/tutorials/tqdm_progress_bar#default_tqdmcallback_usage
# - [x] change `Manager`s path from `test_` to real
# - [x] improve shuffling... we can see that data is almost contiguous
# - [x] calculate heat flux and use it instead of power
# - [ ] see papers in Kopernio
# - [x] when describing a dataset, include only the DictImageTransformer components that appear as
#     ExperimentVideos
# - [ ] allow bootstraping for confidence intervals in `bl.model.eval_with`
# - [ ] define function `eval_regions` that allows us to evaluate a model when
#     nominal power = 0, 10, 20... for instance:
#     `eval_regions(model, ds_val, 'nominal_heat_flux') == {0: 8.32, 10: 5.63, 20: 12.10, ...}`
# """
