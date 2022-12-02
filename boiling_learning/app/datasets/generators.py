from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, TypeAlias

import funcy
import modin.pandas as pd
import numpy as np

from boiling_learning.app import options
from boiling_learning.app.datasets.raw.boiling1d import BOILING_CASES
from boiling_learning.app.datasets.raw.condensation import CONDENSATION_DATASETS
from boiling_learning.app.paths import ANALYSES_PATH
from boiling_learning.datasets.cache import NumpyCache
from boiling_learning.datasets.datasets import DatasetSplits, DatasetTriplet
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.image_datasets import Image, ImageDataset, ImageDatasetTriplet, Targets
from boiling_learning.io.storage import dataclass
from boiling_learning.lazy import LazyDescribed, eager
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.scripts import set_boiling_cases_data
from boiling_learning.transforms import map_transformers

_IsCondensation: TypeAlias = bool

BOILING_VIDEO_TO_SETTER = {
    video: partial(set_boiling_cases_data.main, case())
    for case in BOILING_CASES
    for video in case()
}

EXTRACTED_FRAMES_DIRECTORY_ALLOCATORS: dict[_IsCondensation, JSONAllocator] = {
    False: JSONAllocator(ANALYSES_PATH / 'datasets' / 'frames' / 'boiling'),
    True: JSONAllocator(ANALYSES_PATH / 'datasets' / 'frames' / 'condensation'),
}

NUMPY_DIRECTORY_ALLOCATORS: dict[_IsCondensation, JSONAllocator] = {
    False: JSONAllocator(ANALYSES_PATH / 'datasets' / 'numpy' / 'boiling'),
    True: JSONAllocator(ANALYSES_PATH / 'datasets' / 'numpy' / 'condensation'),
}


def get_image_dataset(
    image_dataset: ExperimentVideoDataset,
    transformers: list[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
    splits: DatasetSplits = DatasetSplits(
        train=Fraction(70, 100),
        val=Fraction(15, 100),
        test=Fraction(15, 100),
    ),
) -> LazyDescribed[ImageDatasetTriplet]:
    purged_experiment_videos = _purge_experiment_videos(image_dataset)

    ds_train_list = []
    ds_val_list = []
    ds_test_list = []
    for video in sorted(image_dataset, key=lambda ev: ev.name):
        if video.name in purged_experiment_videos:
            dataset = _sliceable_dataset_from_video_and_transformers(video, transformers)
            ev_train, ev_val, ev_test = dataset.split(splits.train, splits.val, splits.test)

            ds_train_list.append(ev_train)
            ds_val_list.append(ev_val)
            ds_test_list.append(ev_test)

    # TODO: re-add memory caching here
    ds_train = SliceableDataset.concatenate(*ds_train_list)  # .cache(MemoryCache())
    ds_val = SliceableDataset.concatenate(*ds_val_list)  # .cache(MemoryCache())
    ds_test = SliceableDataset.concatenate(*ds_test_list)  # .cache(MemoryCache())

    return LazyDescribed.from_value_and_description(
        DatasetTriplet(ds_train, ds_val, ds_test),
        {
            'image_dataset': image_dataset,
            'transformers': transformers,
            'splits': splits,
        },
    )


def compile_transformers(
    transformers: Iterable[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
    experiment_video: ExperimentVideo,
) -> LazyDescribed[Callable[[Image], Image]]:
    compiled_transformers = tuple(
        transformer[experiment_video.name] if isinstance(transformer, dict) else transformer
        for transformer in transformers
    )
    return LazyDescribed.from_value_and_description(
        funcy.rcompose(*compiled_transformers),
        compiled_transformers,
    )


@cache(JSONAllocator(ANALYSES_PATH / 'cache' / 'purged-experiment-videos'))
def _purge_experiment_videos(image_dataset: ExperimentVideoDataset) -> list[str]:
    return [video.name for video in tuple(image_dataset) if _ensure_data_is_set(video)]


def _ensure_data_is_set(video: ExperimentVideo) -> bool:
    if video.data is None:
        setter = BOILING_VIDEO_TO_SETTER[video]
        setter()

    return video.data is not None


def _sliceable_dataset_from_video_and_transformers(
    ev: ExperimentVideo,
    transformers: Iterable[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
) -> ImageDataset:
    _ensure_data_is_set(ev)
    video = _video_dataset_from_video_and_transformers(ev, transformers)
    targets = _target_dataset_from_video(ev)

    # return SliceableDataset.zip(video, targets, strictness='one-off')
    # return SliceableDataset.zip(video, targets, strictness="none")
    return SliceableDataset.zip(
        video,
        targets,
        strictness='none' if _is_condensation_video(ev) else 'one-off',
    )


def _target_dataset_from_video(video: ExperimentVideo) -> SliceableDataset[Targets]:
    targets = _experiment_video_targets_as_dataframe(video)
    return SliceableDataset.from_sequence(targets.to_dict('records'))


def _experiment_video_targets_as_dataframe(video: ExperimentVideo) -> pd.DataFrame:
    return (
        _experiment_video_targets_as_dataframe_condensation
        if _is_condensation_video(video)
        else _experiment_video_targets_as_dataframe_boiling
    )(video)


def _dataframe_targets_to_csv(targets: pd.DataFrame, path: Path) -> None:
    targets.to_csv(path, index=False)


@cache(
    JSONAllocator(ANALYSES_PATH / 'datasets' / 'targets' / 'boiling', suffix='.csv'),
    saver=_dataframe_targets_to_csv,
    loader=pd.read_csv,
    exceptions=(OSError, AttributeError),
)
def _experiment_video_targets_as_dataframe_boiling(video: ExperimentVideo) -> pd.DataFrame:
    return video.targets()


@cache(
    JSONAllocator(ANALYSES_PATH / 'datasets' / 'targets' / 'condensation', suffix='.csv'),
    saver=_dataframe_targets_to_csv,
    loader=pd.read_csv,
    exceptions=(OSError, AttributeError),
)
def _experiment_video_targets_as_dataframe_condensation(video: ExperimentVideo) -> pd.DataFrame:
    return video.targets()


def _video_dataset_from_video_and_transformers(
    experiment_video: ExperimentVideo,
    transformers: Iterable[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
) -> SliceableDataset[Image]:
    compiled_transformers = compile_transformers(transformers, experiment_video)

    if options.EXTRACT_FRAMES:
        extracted_frames_directory = EXTRACTED_FRAMES_DIRECTORY_ALLOCATORS[
            _is_condensation_video(experiment_video)
        ].allocate(experiment_video)

        frames = experiment_video.extract_frames(extracted_frames_directory)
    else:
        frames = experiment_video.frames()

    frames = LazyDescribed.from_value_and_description(frames, experiment_video) | map_transformers(
        compiled_transformers
    )
    video_info = _get_video_info(frames)

    numpy_cache_directory = NUMPY_DIRECTORY_ALLOCATORS[
        _is_condensation_video(experiment_video)
    ].allocate(frames)

    return frames().cache(
        NumpyCache(
            numpy_cache_directory,
            shape=(video_info.length, *video_info.shape),
            dtype=np.dtype(video_info.dtype),
        )
    )


@dataclass
class VideoInfo:
    length: int
    shape: tuple[int, ...]
    dtype: str


@cache(JSONAllocator(ANALYSES_PATH / 'cache' / 'video-info'))
@eager
def _get_video_info(video: SliceableDataset[Image]) -> VideoInfo:
    first_frame = video[0]
    return VideoInfo(
        length=len(video),
        shape=first_frame.shape,
        dtype=str(first_frame.dtype),
    )


def _is_condensation_video(ev: ExperimentVideo) -> _IsCondensation:
    return any(ev in dataset() for dataset in CONDENSATION_DATASETS)
