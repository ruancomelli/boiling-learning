import re
import time
from datetime import timedelta
from typing import Dict, Iterable, Optional, Tuple

import gspread
import modin.pandas as pd
import parse
import yaml
from loguru import logger
from oauth2client.client import GoogleCredentials
from sklearn.linear_model import LinearRegression

from boiling_learning.io import json
from boiling_learning.management.allocators import default_table_allocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.preprocessing.video import get_fps
from boiling_learning.utils import KeyedDefaultDict, PathLike, resolve

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
    return timedelta(hours=int(m['h']), minutes=int(m['min']), seconds=int(m['s']))


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

    df['datetime'] = pd.to_datetime(df.pop('date').astype(str) + ' ' + df.pop('time').astype(str))
    datetime_secs = df.datetime.apply(lambda dt: time.mktime(dt.timetuple()))
    df['elapsed_time'] = datetime_secs - datetime_secs.min()
    mass = pd.to_numeric(df.pop('mass [g]').str.replace(',', '.'))
    df['mass'] = mass - mass.min()

    return df


def main(
    datasets: Iterable[ImageDataset],
    dataspecpath: PathLike,
    spreadsheet_name: Optional[str] = None,
    fps_cache_path: Optional[PathLike] = None,
) -> Tuple[ImageDataset, ...]:
    logger.info('Setting condensation data')

    dataspecpath = resolve(dataspecpath)
    dataspec = yaml.safe_load(dataspecpath.read_text())

    if fps_cache_path is not None:
        fps_cache_path = resolve(fps_cache_path)
        allocator = default_table_allocator(fps_cache_path)
        cacher = cache(allocator, saver=json.dump, loader=json.load)
        fps_getter = cacher(get_fps)
    else:
        fps_getter = get_fps

    datasets_dict = KeyedDefaultDict(ImageDataset)
    for dataset in datasets:
        logger.debug(f'Reading condensation dataset {dataset.name}')

        for ev_name, ev in dataset.items():
            case, subcase, test_name, video_name = ev_name.split(':')

            # TODO: add average mass rate to categories?

            logger.debug(f'Setting categories for EV "{ev_name}"')
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

            videospec = dataspec['cases'][case]['subcases'][subcase]['tests'][test_name]['videos'][
                f'{video_name}.mp4'
            ]

            logger.debug(f'Getting FPS for EV "{ev_name}"')

            fps = fps_getter(ev.path)

            logger.debug(f'Getting video data for EV "{ev_name}"')

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

            logger.debug(f'Setting video data for EV "{ev_name}"')
            ev.set_video_data(videodata)

            dataset_name = ':'.join((case, subcase))
            datasets_dict[dataset_name].add(ev)

    grouped_datasets = tuple(datasets_dict.values())

    for dataset in grouped_datasets:
        try:
            logger.debug(f'Trying to load data for {dataset.name}')
            dataset.load_dfs(overwrite=False, missing_ok=False)
            logger.debug(f'Succesfully loaded data for {dataset.name}')
        except FileNotFoundError:
            logger.debug(f'Failed to load data, making dataframes for {dataset.name}')

            dataset.make_dataframe(
                recalculate=False,
                exist_load=True,
                enforce_time=True,
                categories_as_int=True,
                inplace=True,
            )

            for ev in dataset.values():
                _check_experiment_video(ev)

            dataset.save_dfs(overwrite=False)

    if spreadsheet_name is not None:
        mass_data: Optional[Dict[str, pd.DataFrame]] = None

        for dataset in grouped_datasets:
            changed = False

            for ev in dataset.values():
                if 'mass_rate' in ev.df.columns:
                    continue

                changed = True

                if mass_data is None and spreadsheet_name is not None:
                    mass_data = dataframes_from_gspread(spreadsheet_name)

                case_name, subcase_name, test_name, _ = ev.name.split(':')
                df = _parse_mass_timeseries(mass_data, case_name, subcase_name, test_name)
                ev.df['mass_rate'] = _calculate_mass_rate(
                    df, elapsed_time_column='elapsed_time', mass_column='mass'
                )

            if changed:
                dataset.save_dfs(overwrite=True)

    logger.debug(f'Condensation datasets: {grouped_datasets}')

    return grouped_datasets


def _calculate_mass_rate(df: pd.DataFrame, *, elapsed_time_column: str, mass_column: str) -> float:
    X = df[elapsed_time_column].values.reshape(-1, 1)
    y = df[mass_column].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)

    return float(linear_regressor.coef_.squeeze())


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
    raise RuntimeError(
        '*set_condensation_datasets_data* cannot be executed as a standalone script yet.'
    )
