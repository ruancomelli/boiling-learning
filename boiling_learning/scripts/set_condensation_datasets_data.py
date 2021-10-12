import re
import time
from datetime import timedelta
from typing import Dict, Iterable, Optional

import gspread
import modin.pandas as pd
import parse
import yaml
from dataclassy import dataclass
from oauth2client.client import GoogleCredentials
from sklearn.linear_model import LinearRegression

from boiling_learning.io.io import load_json, save_json
from boiling_learning.management.allocators import default_table_allocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.ImageDataset import ImageDataset
from boiling_learning.preprocessing.video import get_fps
from boiling_learning.utils.utils import (
    KeyedDefaultDict,
    PathLike,
    ensure_resolved,
    print_header,
    print_verbose,
)

_subcase_patterns = {
    'T_inf': parse.compile('T_inf {:d}C'),
    'T_s': parse.compile('T_s {:d}C'),
    'rh': parse.compile('rh {:d}%'),
}
_timedelta_pattern = re.compile(r'(?P<h>\d{2}):(?P<min>\d{2}):(?P<s>\d{2})')


def _parse_timedelta(s: Optional[str]) -> Optional[timedelta]:
    if s is None:
        return None

    m = _timedelta_pattern.fullmatch(s)
    return timedelta(
        hours=int(m['h']), minutes=int(m['min']), seconds=int(m['s'])
    )


def dataframes_from_gspread(
    spreadsheet_name: str, credentials: Optional[GoogleCredentials] = None
) -> Dict[str, pd.DataFrame]:
    if credentials is None:
        credentials = GoogleCredentials.get_application_default()

    gc = gspread.authorize(credentials)
    spreadsheet: gspread.Spreadsheet = gc.open(spreadsheet_name)

    return {
        worksheet.title: pd.DataFrame(worksheet.get_all_values())
        for worksheet in spreadsheet.worksheets()
    }


def _parse_mass_timeseries(
    dfs: Dict[str, pd.DataFrame], case: str, subcase: str, test: str
) -> pd.DataFrame:
    df = dfs[case]
    df = df.loc[:, (df.loc[0] == subcase) & (df.loc[1] == test)][2:]
    df, df.columns = df[1:], df.iloc[0]

    df['datetime'] = pd.to_datetime(
        df.pop('date').astype(str) + ' ' + df.pop('time').astype(str)
    )
    datetime_secs = df.datetime.apply(lambda dt: time.mktime(dt.timetuple()))
    df['elapsed_time'] = datetime_secs - datetime_secs.min()
    mass = pd.to_numeric(df.pop('mass [g]').str.replace(',', '.'))
    df['mass'] = mass - mass.min()

    return df


@dataclass
class LinearRegressionCoefficients:
    coef: float
    intercept: float


def linear_regression(
    df: pd.DataFrame, x_column: str, y_column: str
) -> LinearRegressionCoefficients:
    X = df[x_column].values.reshape(-1, 1)
    y = df[y_column].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)

    return LinearRegressionCoefficients(
        coef=float(linear_regressor.coef_.squeeze()),
        intercept=float(linear_regressor.intercept_.squeeze()),
    )


def main(
    datasets: Iterable[ImageDataset],
    dataspecpath: PathLike,
    spreadsheet_name: Optional[str] = None,
    verbose: int = 0,
    fps_cache_path: Optional[PathLike] = None,
) -> Dict[str, ImageDataset]:
    dataspecpath = ensure_resolved(dataspecpath)
    dataspec = yaml.safe_load(dataspecpath.read_text())

    if fps_cache_path is not None:

        def verbose_save(obj, path):
            print('Saving...')
            print('> obj:', obj)
            print('> path:', path)
            return save_json(obj, path)

        def verbose_load(path):
            print('Loading from', path)
            return load_json(path)

        fps_cache_path = ensure_resolved(fps_cache_path)
        allocator = default_table_allocator(fps_cache_path)
        cacher = cache(allocator, saver=verbose_save, loader=verbose_load)
        fps_getter = cacher(get_fps)
    else:
        fps_getter = get_fps

    datasets_dict = KeyedDefaultDict(ImageDataset)
    for dataset in datasets:
        print_verbose(verbose >= 2, f'Reading dataset {dataset.name}')

        for ev_name, ev in dataset.items():
            case, subcase, test_name, video_name = ev_name.split(':')

            # TODO: add average mass rate to categories?

            print_verbose(
                verbose >= 2, f'Setting categories for EV "{ev_name}"'
            )
            if case == 'stainless steel' and subcase == 'polished':
                # those are the standard conditions for polished SS
                # temperatures are in Celsius
                # relative humidity is in percentage
                # age can be either "new" or "old"
                categories = {'age': 'new', 'T_inf': 50, 'T_s': 10, 'rh': 80}
            elif case == 'parametric':
                if subcase == 'old':
                    categories = {
                        'age': 'old',
                        'T_inf': 50,
                        'T_s': 10,
                        'rh': 80,
                    }
                else:
                    categories = {
                        'age': 'new',
                        'T_inf': 50,
                        'T_s': 10,
                        'rh': 80,
                    }
                    for change, pattern in _subcase_patterns.items():
                        match = pattern.search(subcase)
                        if match is not None:
                            categories[change] = match[0]
                            break
            else:
                continue

            videospec = dataspec['cases'][case]['subcases'][subcase]['tests'][
                test_name
            ]['videos'][video_name + '.mp4']

            print_verbose(verbose >= 2, f'Getting FPS for EV "{ev_name}"')

            fps = fps_getter(ev.path)

            print_verbose(
                verbose >= 2, f'Getting video data for EV "{ev_name}"'
            )

            videodata = ExperimentVideo.VideoData(
                categories=categories,
                fps=fps,
                # since there is no syncing between video and experimental data here,
                # we simply set the first frame as the reference
                ref_index=0,
                ref_elapsed_time=0,
                start_elapsed_time=_parse_timedelta(videospec['start']),
                end_elapsed_time=_parse_timedelta(videospec['end']),
            )

            print_verbose(
                verbose >= 2, f'Setting video data for EV "{ev_name}"'
            )
            ev.set_video_data(videodata)
            print_verbose(
                verbose, f'{ev_name} -> [{ev.start}, {ev.end}) :: {categories}'
            )

            dataset_name = ':'.join((case, subcase))
            datasets_dict[dataset_name].add(ev)

    for dataset in datasets_dict.values():
        if verbose:
            print_header(dataset.name)

        try:
            print_verbose(verbose, 'Loading')
            dataset.load_dfs(overwrite=False, missing_ok=False)
        except FileNotFoundError:
            print_verbose(verbose, 'Failed, making dataframes.')

            dataset.make_dataframe(
                recalculate=False,
                exist_load=True,
                enforce_time=True,
                categories_as_int=True,
                inplace=True,
            )

            for ev in dataset.values():
                _indices = tuple(map(int, ev.df[ev.column_names.index]))
                _expected = tuple(range(len(ev.video)))
                if _indices != _expected:
                    raise ValueError(
                        f'expected indices != indices for {ev.name}.'
                        f' Got expected: {_expected}'
                        f' Got indices: {_indices}'
                    )

            dataset.save_dfs(overwrite=False)

    if spreadsheet_name is not None:
        MASS_COLUMN: str = 'mass_rate'
        mass_data: Optional[Dict[str, pd.DataFrame]] = None

        for dataset in datasets_dict.values():
            changed: bool = False

            for ev in dataset.values():
                if MASS_COLUMN in ev.df.columns:
                    continue

                changed = True

                if mass_data is None and spreadsheet_name is not None:
                    mass_data = dataframes_from_gspread(spreadsheet_name)

                case_name, subcase_name, test_name, _ = ev.name.split(':')
                df = _parse_mass_timeseries(
                    mass_data, case_name, subcase_name, test_name
                )
                coefs = linear_regression(df, 'elapsed_time', 'mass')
                ev.df[MASS_COLUMN] = coefs.coef

            if changed:
                dataset.save_dfs(overwrite=True)

    print_verbose(verbose, 'Condensation datasets dict:')
    for ds_name, ds in datasets_dict.items():
        print_verbose(verbose, ds_name, '::', ds)

    return datasets_dict


if __name__ == '__main__':
    raise RuntimeError(
        '*set_condensation_datasets_data* cannot be executed as a standalone script yet.'
    )
