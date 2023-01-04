import functools
from pathlib import Path
from typing import Literal

import modin.pandas as pd
import numpy as np
from loguru import logger

from boiling_learning.app.paths import data_path, shared_cache_path
from boiling_learning.data.samples import WIRE_SAMPLES
from boiling_learning.io.storage import load, save
from boiling_learning.lazy import LazyCallable, LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.experiment_video import ExperimentVideo, VideoData
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.utils.pathutils import PathLike, resolve
from boiling_learning.utils.printing import add_unit_post_fix
from boiling_learning.utils.units import unit_registry as ureg


def boiling_data_path() -> Path:
    return data_path() / 'boiling1d'


@functools.cache
def boiling_cases() -> tuple[LazyDescribed[ExperimentVideoDataset], ...]:
    return tuple(
        _case_from_path(boiling_data_path() / case_name)
        for case_name in ('case 1', 'case 2', 'case 3', 'case 4')
    )


def _case_from_path(path: PathLike, /) -> LazyDescribed[ExperimentVideoDataset]:
    return LazyDescribed(
        _load_case_from_path(path),
        # type-ignore is necessary until the classes plugin works again
        path,  # type: ignore[arg-type]
    )


@LazyCallable
def _load_case_from_path(path: PathLike, /) -> ExperimentVideoDataset:
    logger.info(f'Loading boiling case from {path}')
    loader = _cached_load_case_from_path(experiment='boiling1d')
    return loader(resolve(path))


def _cached_load_case_from_path(*, experiment: Literal['boiling1d', 'condensation']):
    @cache(
        JSONAllocator(shared_cache_path() / 'experiment-video-datasets' / experiment),
        saver=custom_experiment_video_dataset_save,
        loader=custom_experiment_video_dataset_load,
    )
    def _load_case_from_path(path: Path, /) -> ExperimentVideoDataset:
        return _set_boiling_case_data(
            Case(path, video_suffix='.MP4').convert_videos(
                '.mp4',
                'converted',
                overwrite=False,
            )
        )

    return _load_case_from_path


def custom_experiment_video_dataset_save(
    experiment_video_dataset: ExperimentVideoDataset, path: PathLike
) -> None:
    path = resolve(path, dir=True)
    save(len(experiment_video_dataset), path / 'length')
    for index, ev in enumerate(experiment_video_dataset):
        custom_experiment_video_save(ev, path / str(index))


def custom_experiment_video_dataset_load(path: PathLike) -> ExperimentVideoDataset:
    path = resolve(path)
    length = load(path / 'length')

    return ExperimentVideoDataset(
        custom_experiment_video_load(path / str(index)) for index in range(length)
    )


def custom_experiment_video_save(ev: ExperimentVideo, path: Path) -> None:
    save(
        {
            'video_path': ev.path,
            'df_path': ev.df_path,
            'name': ev.name,
        },
        path / 'properties',
    )
    save(ev.data, path / 'video_data')
    ev.df.to_csv(path / 'data.csv', index=False)


def custom_experiment_video_load(path: Path) -> ExperimentVideo:
    properties = load(path / 'properties')
    data = load(path / 'video_data')
    df = pd.read_csv(path / 'data.csv')

    return ExperimentVideo(
        properties['video_path'], properties['df_path'], properties['name'], data, df
    )


def _set_boiling_case_data(case: Case, /) -> ExperimentVideoDataset:
    logger.info(f'Setting boiling data for case {case.name}')

    dataset = case.set_video_data_from_file()

    for ev in dataset:
        try:
            logger.debug(f'Trying to load data for {ev.name}')
            ev.df = ev.load_df()
            logger.debug(f'Succesfully loaded data for {ev.name}')
        except FileNotFoundError:
            logger.debug(f'Failed to load data for {ev.name}')

            _set_experiment_video_data(ev, case.get_experimental_data())

    return dataset


def _set_experiment_video_data(ev: ExperimentVideo, source: pd.DataFrame) -> None:
    dataframe = ev.sync_time_series(source)

    assert ev.data is not None
    ev.df = _regularize_experiment_video_dataframe(dataframe, ev.data)

    ev.save_df(ev.df)


def _regularize_experiment_video_dataframe(
    dataframe: pd.DataFrame, data: VideoData
) -> pd.DataFrame:
    dataframe = dataframe.drop(
        columns=[
            'Time instant',
            'Flux [W/m^2]',
            'Flux [W/cm^2]',
            'Bulk Temperature [deg C]',
            'Wire Temperature [deg C]',
            'Superheat [deg C]',
        ],
        errors='ignore',
    )

    power_unit = ureg.watt
    heat_flux_unit = ureg.watt / ureg.centimeter**2
    sample_id = data.categories['sample_id']

    full_power_key = add_unit_post_fix('Power', power_unit)
    full_heat_flux_key = add_unit_post_fix('Flux', heat_flux_unit)

    lateral_area = WIRE_SAMPLES[sample_id].lateral_area()
    power = np.array(dataframe[full_power_key]) * power_unit
    flux = power / lateral_area

    dataframe[full_heat_flux_key] = flux.to(heat_flux_unit).magnitude

    return dataframe
