import re
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
from boiling_learning.preprocessing.video import get_fps, valid_end_frame
from boiling_learning.scripts.utils.setting_data import check_experiment_video_dataframe_indices
from boiling_learning.utils import KeyedDefaultDict, PathLike, resolve
from boiling_learning.utils.frozendict import frozendict

_SUBCASE_PATTERNS = frozendict(
    {
        'T_inf': parse.compile('T_inf {:d}C'),
        'T_s': parse.compile('T_s {:d}C'),
        'rh': parse.compile('rh {:d}%'),
    }
)
_TIMEDELTA_PATTERN = re.compile(r'(?P<h>\d{2}):(?P<min>\d{2}):(?P<s>\d{2})')


@lru_cache(maxsize=None)
def dataframes_from_gspread(
    spreadsheet_name: str, credentials: Optional[GoogleCredentials] = None
) -> Dict[str, pd.DataFrame]:
    if credentials is None:
        credentials = GoogleCredentials.get_application_default()

    gc = gspread.authorize(credentials)
    spreadsheet = gc.open(spreadsheet_name)

    return {
        worksheet.title: pd.DataFrame(worksheet.get_all_values())
        for worksheet in spreadsheet.worksheets()
    }


def main(
    datasets: Iterable[ImageDataset],
    dataspecpath: PathLike,
    spreadsheet_name: Optional[str] = None,
    *,
    fps_cache_path: Optional[PathLike] = None,
    end_frame_index_cache_path: Optional[PathLike] = None,
) -> Tuple[ImageDataset, ...]:
    logger.info('Setting condensation data')

    datasets = tuple(datasets)
    dataspecpath = resolve(dataspecpath)
    dataspec = yaml.safe_load(dataspecpath.read_text())

    fps_getter = _generate_fps_getter(fps_cache_path)
    end_frame_index_getter = _generate_end_frame_index_getter(end_frame_index_cache_path)

    for dataset in datasets:
        _set_dataset_data(
            dataset, dataspec, fps_getter=fps_getter, end_frame_index_getter=end_frame_index_getter
        )

    grouped_datasets = _group_datasets(datasets)

    for dataset in grouped_datasets:
        _make_dataframe(dataset)

    if spreadsheet_name is not None:
        _set_mass_rate(grouped_datasets, spreadsheet_name)

    logger.debug(f'Condensation datasets: {grouped_datasets}')

    return grouped_datasets


def _generate_fps_getter(fps_cache_path: Optional[PathLike]) -> Callable[[Path], float]:
    if fps_cache_path is None:
        return get_fps

    allocator = default_table_allocator(fps_cache_path)
    cacher = cache(allocator, saver=json.dump, loader=json.load)
    return cacher(get_fps)


def _generate_end_frame_index_getter(
    end_frame_index_cache_path: Optional[PathLike],
) -> Callable[[ExperimentVideo], int]:
    if end_frame_index_cache_path is None:
        return valid_end_frame

    allocator = default_table_allocator(end_frame_index_cache_path)
    cacher = cache(allocator, saver=json.dump, loader=json.load)
    return cacher(valid_end_frame)


def _set_mass_rate(grouped_datasets: Tuple[ImageDataset, ...], spreadsheet_name: str) -> None:
    for dataset in grouped_datasets:
        for ev in dataset:
            if 'mass_rate' in ev.df.columns:
                continue

            mass_data = dataframes_from_gspread(spreadsheet_name)
            case_name, subcase_name, test_name, _ = ev.name.split(':')
            df = _parse_mass_timeseries(mass_data, case_name, subcase_name, test_name)
            ev.df['mass_rate'] = _calculate_mass_rate(
                df, elapsed_time_column='elapsed_time', mass_column='mass'
            )
            ev.save_df(overwrite=True)


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


def _set_dataset_data(
    dataset: ImageDataset,
    dataspec: Dict[str, Any],
    *,
    fps_getter: Callable[[Path], float],
    end_frame_index_getter: Callable[[ExperimentVideo], int],
) -> None:
    logger.debug(f'Reading condensation dataset {dataset.name}')

    for ev in dataset:
        _set_ev_data(
            ev, dataspec, fps_getter=fps_getter, end_frame_index_getter=end_frame_index_getter
        )


def _set_ev_data(
    ev: ExperimentVideo,
    dataspec: Dict[str, Any],
    *,
    fps_getter: Callable[[Path], float],
    end_frame_index_getter: Callable[[ExperimentVideo], int],
) -> None:
    case, subcase, test_name, video_name = ev.name.split(':')

    # TODO: add average mass rate to categories?

    logger.debug(f'Setting categories for EV "{ev.name}"')
    if case == 'stainless steel' and subcase == 'polished':
        # those are the standard conditions for polished SS
        # temperatures are in Celsius
        # relative humidity is in percentage
        # age can be either "new" or "old"
        categories = {'age': 'new', 'T_inf': 50, 'T_s': 10, 'rh': 80}
    elif case == 'parametric':
        if subcase == 'old':
            categories = {'age': 'old', 'T_inf': 50, 'T_s': 10, 'rh': 80}
        else:
            categories = {'age': 'new', 'T_inf': 50, 'T_s': 10, 'rh': 80}
            for change, pattern in _SUBCASE_PATTERNS.items():
                match = pattern.search(subcase)
                if match is not None:
                    categories[change] = match[0]
                    break
    else:
        return

    videospec = dataspec['cases'][case]['subcases'][subcase]['tests'][test_name]['videos'][
        f'{video_name}.mp4'
    ]

    logger.debug(f'Getting FPS for EV "{ev.name}"')

    fps = fps_getter(ev.path)

    logger.debug(f'Getting video data for EV "{ev.name}"')

    videodata = ExperimentVideo.VideoData(
        categories=categories,
        fps=fps,
        # since there is no syncing between video and experimental data here,
        # we simply set the first frame as the reference
        ref_index=0,
        ref_elapsed_time=timedelta(0),
        start_elapsed_time=_parse_timedelta(videospec['start']),
        end_elapsed_time=_parse_timedelta(videospec['end']),
    )

    logger.debug(f'Setting video data for EV "{ev.name}"')

    end_frame_index = end_frame_index_getter(ev)
    ev.video = ev.video[: end_frame_index + 1]
    ev.data = videodata


def _parse_timedelta(s: Optional[str]) -> Optional[timedelta]:
    if s is None:
        return None

    m = _TIMEDELTA_PATTERN.fullmatch(s)

    if m is None:
        return None

    return timedelta(hours=int(m['h']), minutes=int(m['min']), seconds=int(m['s']))


def _make_dataframe(dataset: ImageDataset) -> None:
    missing: List[ExperimentVideo] = []
    for ev in dataset:
        try:
            logger.debug(f'Trying to load data for {ev.name}')
            ev.load_df(overwrite=False, missing_ok=False, inplace=True)
            logger.debug(f'Succesfully loaded data for {ev.name}')
        except FileNotFoundError:
            logger.debug(f'Failed to load data for {ev.name}')
            missing.append(ev)

    for ev in missing:
        logger.debug(f'Making dataframe for {ev.name}')

        ev.make_dataframe(
            recalculate=False,
            exist_load=True,
            enforce_time=True,
            categories_as_int=True,
            inplace=True,
        )

        check_experiment_video_dataframe_indices(ev)

        ev.save_df(overwrite=False)


def _group_datasets(datasets: Iterable[ImageDataset]) -> Tuple[ImageDataset, ...]:
    datasets_dict = KeyedDefaultDict[str, ImageDataset](ImageDataset)
    for dataset in datasets:
        logger.debug(f'Reading condensation dataset {dataset.name}')

        for ev in dataset:
            # avoid including experiment videos that didn't have their data correctly
            # set
            if ev.data is not None:
                case, subcase, _test_name, _video_name = ev.name.split(':')
                dataset_name = ':'.join((case, subcase))
                datasets_dict[dataset_name].add(ev)

    return tuple(datasets_dict.values())


def _calculate_mass_rate(df: pd.DataFrame, *, elapsed_time_column: str, mass_column: str) -> float:
    X = df[elapsed_time_column].values.reshape(-1, 1)
    y = df[mass_column].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)

    return float(linear_regressor.coef_.squeeze())


if __name__ == '__main__':
    raise RuntimeError(
        '*set_condensation_datasets_data* cannot be executed as a standalone script yet.'
    )
