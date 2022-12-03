import functools
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Literal

import funcy
import modin.pandas as pd
import numpy as np

from boiling_learning.app import options
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import analyses_path
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


def get_image_dataset(
    image_dataset: ExperimentVideoDataset,
    transformers: list[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
    *,
    splits: DatasetSplits = DatasetSplits(
        train=Fraction(70, 100),
        val=Fraction(15, 100),
        test=Fraction(15, 100),
    ),
    experiment: Literal['boiling1d', 'condensation'],
) -> LazyDescribed[ImageDatasetTriplet]:
    purged_experiment_videos = _experiment_video_purger(experiment=experiment)(image_dataset)

    ds_train_list = []
    ds_val_list = []
    ds_test_list = []
    for video in sorted(image_dataset, key=lambda ev: ev.name):
        if video.name in purged_experiment_videos:
            dataset = _sliceable_dataset_from_video_and_transformers(
                video,
                transformers,
                experiment=experiment,
            )
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


def _experiment_video_purger(
    *, experiment: Literal['boiling1d', 'condensation']
) -> Callable[[ExperimentVideoDataset], list[str]]:
    @cache(JSONAllocator(analyses_path() / 'cache' / 'purged-experiment-videos'))
    def _purge_experiment_videos(image_dataset: ExperimentVideoDataset) -> list[str]:
        return [
            video.name
            for video in tuple(image_dataset)
            if _ensure_data_is_set(video, experiment=experiment)
        ]

    return _purge_experiment_videos


def _ensure_data_is_set(
    video: ExperimentVideo,
    experiment: Literal['boiling1d', 'condensation'],
) -> bool:
    if video.data is None and experiment == 'boiling1d':
        case = next(case() for case in boiling_cases() if video in case())
        set_boiling_cases_data.main(case)

    return video.data is not None


def _sliceable_dataset_from_video_and_transformers(
    ev: ExperimentVideo,
    transformers: Iterable[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
    *,
    experiment: Literal['boiling1d', 'condensation'],
) -> ImageDataset:
    _ensure_data_is_set(ev, experiment=experiment)
    video = _video_dataset_from_video_and_transformers(ev, transformers, experiment=experiment)
    targets = _target_dataset_from_video(ev, experiment=experiment)

    # return SliceableDataset.zip(video, targets, strictness='one-off')
    # return SliceableDataset.zip(video, targets, strictness="none")
    return SliceableDataset.zip(
        video,
        targets,
        strictness='one-off' if experiment == 'boiling1d' else 'none',
    )


def _target_dataset_from_video(
    video: ExperimentVideo,
    *,
    experiment: Literal['boiling1d', 'condensation'],
) -> SliceableDataset[Targets]:
    targets = _experiment_video_targets_as_dataframe(video, experiment=experiment)
    return SliceableDataset.from_sequence(targets.to_dict('records'))


def _experiment_video_targets_as_dataframe(
    video: ExperimentVideo,
    *,
    experiment: Literal['boiling1d', 'condensation'],
) -> pd.DataFrame:
    return _experiment_video_target_getter(experiment)(video)


@functools.cache
def _experiment_video_target_getter(
    experiment: Literal['boiling1d', 'condensation'],
) -> Callable[[ExperimentVideo], pd.DataFrame]:
    @cache(
        JSONAllocator(analyses_path() / 'datasets' / 'targets' / experiment, suffix='.csv'),
        saver=_dataframe_targets_to_csv,
        loader=pd.read_csv,
        exceptions=(OSError, AttributeError),
    )
    def _target_getter(video: ExperimentVideo) -> pd.DataFrame:
        return video.targets()

    return _target_getter


def _dataframe_targets_to_csv(targets: pd.DataFrame, path: Path) -> None:
    targets.to_csv(path, index=False)


def _video_dataset_from_video_and_transformers(
    experiment_video: ExperimentVideo,
    transformers: Iterable[Transformer[Image, Image] | dict[str, Transformer[Image, Image]]],
    *,
    experiment: Literal['boiling1d', 'condensation'],
) -> SliceableDataset[Image]:
    compiled_transformers = compile_transformers(transformers, experiment_video)

    if options.EXTRACT_FRAMES and experiment == 'boiling1d':
        extracted_frames_directory = _extracted_frames_directory_allocator(experiment).allocate(
            experiment_video
        )

        frames = experiment_video.extract_frames(extracted_frames_directory)
    else:
        frames = experiment_video.frames()

    frames = LazyDescribed.from_value_and_description(frames, experiment_video) | map_transformers(
        compiled_transformers
    )
    video_info = _video_info_getter()(frames)

    numpy_cache_directory = _numpy_directory_allocator(experiment).allocate(frames)

    return frames().cache(
        NumpyCache(
            numpy_cache_directory,
            shape=(video_info.length, *video_info.shape),
            dtype=np.dtype(video_info.dtype),
        )
    )


@functools.cache
def _extracted_frames_directory_allocator(
    experiment: Literal['boiling1d', 'condensation']
) -> JSONAllocator:
    return JSONAllocator(analyses_path() / 'datasets' / 'frames' / experiment)


@functools.cache
def _numpy_directory_allocator(experiment: Literal['boiling1d', 'condensation']) -> JSONAllocator:
    return JSONAllocator(analyses_path() / 'datasets' / 'numpy' / experiment)


@dataclass
class VideoInfo:
    length: int
    shape: tuple[int, ...]
    dtype: str


def _video_info_getter() -> Callable[[SliceableDataset[Image]], VideoInfo]:
    @cache(JSONAllocator(analyses_path() / 'cache' / 'video-info'))
    @eager
    def _get_video_info(video: SliceableDataset[Image]) -> VideoInfo:
        first_frame = video[0]
        return VideoInfo(
            length=len(video),
            shape=first_frame.shape,
            dtype=str(first_frame.dtype),
        )

    return _get_video_info
