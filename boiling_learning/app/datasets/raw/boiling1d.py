from functools import cache
from pathlib import Path

import modin.pandas as pd
import numpy as np
from loguru import logger

from boiling_learning.app.paths import data_path
from boiling_learning.data.samples import WIRE_SAMPLES
from boiling_learning.lazy import LazyCallable, LazyDescribed
from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.scripts.utils.setting_data import check_experiment_video_dataframe_indices
from boiling_learning.utils.pathutils import PathLike
from boiling_learning.utils.printing import add_unit_post_fix
from boiling_learning.utils.units import unit_registry as ureg


def boiling_data_path() -> Path:
    return data_path() / 'boiling1d'


@cache
def boiling_cases() -> tuple[LazyDescribed[Case], ...]:
    return tuple(
        _case_from_path(boiling_data_path() / case_name)
        for case_name in ('case 1', 'case 2', 'case 3', 'case 4')
    )


def _case_from_path(path: PathLike, /) -> LazyDescribed[Case]:
    return LazyDescribed(
        _load_case_from_path(path),
        # type-ignore is necessary until the classes plugin works again
        path,  # type: ignore[arg-type]
    )


@LazyCallable
def _load_case_from_path(path: PathLike, /) -> Case:
    logger.info(f'Loading boiling case from {path}')
    return _set_boiling_case_data(
        Case(path, video_suffix='.MP4').convert_videos(
            '.mp4',
            'converted',
            overwrite=False,
        )
    )


def _set_boiling_case_data(case: Case, /) -> Case:
    logger.info(f'Setting boiling data for case {case.name}')

    case.set_video_data_from_file(remove_absent=True)

    for ev in case:
        try:
            logger.debug(f'Trying to load data for {ev.name}')
            ev.load_df(overwrite=False, inplace=True)
            logger.debug(f'Succesfully loaded data for {ev.name}')
        except FileNotFoundError:
            logger.debug(f'Failed to load data for {ev.name}')

            _set_experiment_video_data(ev, case.get_experimental_data())

    return case


def _set_experiment_video_data(ev: ExperimentVideo, df: pd.DataFrame) -> None:
    ev.df = ev.sync_time_series(df)
    check_experiment_video_dataframe_indices(ev)
    ev.df = _regularize_experiment_video_dataframe(ev)
    ev.save_df()


def _regularize_experiment_video_dataframe(ev: ExperimentVideo) -> pd.DataFrame:
    assert ev.df is not None
    assert ev.data is not None

    df = ev.df.drop(
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
    sample_id = ev.data.categories['sample_id']

    full_power_key = add_unit_post_fix('Power', power_unit)
    full_heat_flux_key = add_unit_post_fix('Flux', heat_flux_unit)

    lateral_area = WIRE_SAMPLES[sample_id].lateral_area()
    power = np.array(df[full_power_key]) * power_unit
    flux = power / lateral_area

    df[full_heat_flux_key] = flux.to(heat_flux_unit).magnitude

    return df
