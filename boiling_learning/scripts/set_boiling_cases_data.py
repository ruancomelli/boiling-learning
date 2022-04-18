from typing import Iterable, Mapping

import numpy as np
from loguru import logger

from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experimental_data import ExperimentalData
from boiling_learning.utils import PathLike, geometry
from boiling_learning.utils.frozendict import frozendict
from boiling_learning.utils.printing import add_unit_post_fix
from boiling_learning.utils.units import unit_registry as ureg

SAMPLES = frozendict(
    {
        1: geometry.Cylinder(length=6.5 * ureg.centimeter, diameter=0.51 * ureg.millimeter),
        2: geometry.Cylinder(length=6.5 * ureg.centimeter, diameter=0.51 * ureg.millimeter),
        3: geometry.Cylinder(length=6.5 * ureg.centimeter, diameter=0.25 * ureg.millimeter),
        4: geometry.RectangularPrism(
            length=6.5 * ureg.centimeter,
            width=1 / 16 * ureg.inch,
            thickness=0.0031 * ureg.inch,
        ),
        5: geometry.RectangularPrism(
            length=6.5 * ureg.centimeter,
            width=1 / 16 * ureg.inch,
            thickness=0.0031 * ureg.inch,
        ),
    }
)


def main(cases: Iterable[Case], *, case_experiment_map: Mapping[str, PathLike]) -> None:
    logger.info('Setting boiling data')

    for case in cases:
        set_case(case, case_experiment_path=case_experiment_map[case.name])


def set_case(case: Case, *, case_experiment_path: PathLike) -> None:
    case.set_video_data_from_file(purge=True, remove_absent=True)

    try:
        logger.debug(f'Trying to load data for {case.name}')
        case.load_dfs(overwrite=False, missing_ok=False)
        logger.debug(f'Succesfully loaded data for {case.name}')
    except FileNotFoundError:
        logger.debug(f'Failed to load data, making dataframes for {case.name}')
        _set_case_data(case, case_experiment_path=case_experiment_path)


def _set_case_data(case: Case, *, case_experiment_path: PathLike) -> None:
    df = (
        ExperimentalData(case_experiment_path)
        .as_dataframe()
        .drop(columns='Time instant')
        .astype({'Elapsed time': 'float64'})
        .set_index('Elapsed time')
    )

    case.sync_time_series(df)
    case.make_dataframe(
        recalculate=False,
        exist_load=False,
        enforce_time=True,
        categories_as_int=True,
        inplace=True,
    )

    for ev in case:
        _check_experiment_video(ev)

    for ev in case:
        _regularize_experiment_video_dataframe(ev)

    case.save_dfs(overwrite=False)


def _regularize_experiment_video_dataframe(ev: ExperimentVideo) -> None:
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

    lateral_area = SAMPLES[sample_id].lateral_area
    power = np.array(ev.df[full_power_key]) * power_unit
    flux = power / lateral_area

    ev.df[full_heat_flux_key] = flux.to(heat_flux_unit).magnitude


def _check_experiment_video(ev: ExperimentVideo) -> None:
    indices = tuple(map(int, ev.df[ev.column_names.index]))
    expected = tuple(range(len(ev.video)))
    if indices != expected:
        raise ValueError(
            f'expected indices != indices for {ev.name}.'
            f' Got expected: {expected}.'
            f' Got indices: {indices}.'
        )


if __name__ == '__main__':
    raise RuntimeError('*set_boiling_cases_data* cannot be executed as a standalone script yet.')
