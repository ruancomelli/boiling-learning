from typing import Iterable

import modin.pandas as pd
import numpy as np
from loguru import logger

from boiling_learning.data.samples import WIRE_SAMPLES
from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.scripts.utils.setting_data import check_experiment_video_dataframe_indices
from boiling_learning.utils.printing import add_unit_post_fix
from boiling_learning.utils.units import unit_registry as ureg


def main(cases: Iterable[Case], /) -> None:
    logger.info('Setting boiling data')

    for case in cases:
        set_case(case)


def set_case(case: Case, /) -> None:
    case.set_video_data_from_file(remove_absent=True)

    for ev in case:
        try:
            logger.debug(f'Trying to load data for {ev.name}')
            ev.load_df(overwrite=False, inplace=True)
            logger.debug(f'Succesfully loaded data for {ev.name}')
        except FileNotFoundError:
            logger.debug(f'Failed to load data for {ev.name}')

            _set_experiment_video_data(ev, case.get_experimental_data())


def _set_experiment_video_data(ev: ExperimentVideo, df: pd.DataFrame) -> None:
    ev.sync_time_series(df, inplace=True)
    ev.make_dataframe(enforce_time=True, inplace=True)
    check_experiment_video_dataframe_indices(ev)
    _regularize_experiment_video_dataframe(ev)
    ev.save_df(overwrite=False)


def _regularize_experiment_video_dataframe(ev: ExperimentVideo) -> None:
    assert ev.df is not None
    assert ev.data is not None

    ev.df = ev.df.drop(
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
    power = np.array(ev.df[full_power_key]) * power_unit
    flux = power / lateral_area

    ev.df[full_heat_flux_key] = flux.to(heat_flux_unit).magnitude
