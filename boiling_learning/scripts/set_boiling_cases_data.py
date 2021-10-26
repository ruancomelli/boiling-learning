from typing import Iterable, Mapping

import numpy as np

from boiling_learning.preprocessing.Case import Case
from boiling_learning.preprocessing.ExperimentalData import SAMPLES, ExperimentalData
from boiling_learning.utils.printing import add_unit_post_fix
from boiling_learning.utils.units import unit_registry as ureg
from boiling_learning.utils.utils import PathLike, print_header, print_verbose


def main(
    cases: Iterable[Case],
    case_experiment_map: Mapping[str, PathLike],
    verbose: bool = False,
) -> None:
    for case in cases:
        if verbose:
            print_header(case.name)

        case.set_video_data_from_file(purge=True, remove_absent=True)

        try:
            print_verbose(verbose, 'Loading')
            case.load_dfs(overwrite=False, missing_ok=False)
        except FileNotFoundError:
            print_verbose(verbose, 'Failed, making dataframes.')

            df = ExperimentalData(data_path=case_experiment_map[case.name]).as_dataframe()

            df = df.drop(columns='Time instant').astype({'Elapsed time': 'float64'})
            df = df.set_index('Elapsed time')

            case.sync_time_series(df, inplace=True)
            case.make_dataframe(
                recalculate=False,
                exist_load=False,
                enforce_time=True,
                categories_as_int=True,
                inplace=True,
            )

            for ev in case.values():
                _indices = tuple(map(int, ev.df[ev.column_names.index]))
                _expected = tuple(range(len(ev.video)))
                if _indices != _expected:
                    raise ValueError(
                        f'expected indices != indices for {ev.name}.'
                        f' Got expected: {_expected}'
                        f' Got indices: {_indices}'
                    )

            for ev in case.values():
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
                heat_flux_unit = ureg.watt / ureg.centimeter ** 2
                sample_id = ev.data.categories['sample_id']

                full_power_key = add_unit_post_fix('Power', power_unit)
                full_heat_flux_key = add_unit_post_fix('Flux', heat_flux_unit)

                lateral_area = SAMPLES[sample_id].lateral_area
                power = np.array(ev.df[full_power_key]) * power_unit
                flux = power / lateral_area

                ev.df[full_heat_flux_key] = flux.to(heat_flux_unit).magnitude

            case.save_dfs(overwrite=False)


if __name__ == '__main__':
    raise RuntimeError('*set_boiling_cases_data* cannot be executed as a standalone script yet.')
