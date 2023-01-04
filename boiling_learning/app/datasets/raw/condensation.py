import re
from collections.abc import Iterator
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import Any, Iterable, Optional

import parse
import yaml
from frozendict import frozendict  # type: ignore[attr-defined]
from loguru import logger

from boiling_learning.app.paths import data_path
from boiling_learning.lazy import LazyCallable, LazyDescribed
from boiling_learning.preprocessing.experiment_video import ExperimentVideo, VideoData
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.preprocessing.video import Video
from boiling_learning.utils.pathutils import PathLike, resolve


def condensation_data_path() -> Path:
    return data_path() / 'condensation'


def condensation_data_spec_path() -> Path:
    return condensation_data_path() / 'data_spec.yaml'


@cache
def condensation_datasets() -> tuple[LazyDescribed[ExperimentVideoDataset], ...]:
    return _set_condensation_datasets_data(
        (
            _experiment_video_dataset_from_case_and_subcase(case_dir.name, subcase_dir)
            for case_dir in condensation_data_path().iterdir()
            if case_dir.is_dir()
            for subcase_dir in case_dir.iterdir()
            if subcase_dir.is_dir()
        ),
        condensation_data_spec_path(),
    )


_SUBCASE_PATTERNS = frozendict(
    {
        'T_inf': parse.compile('T_inf {:d}C'),
        'T_s': parse.compile('T_s {:d}C'),
        'rh': parse.compile('rh {:d}%'),
    }
)
_TIMEDELTA_PATTERN = re.compile(r'(?P<h>\d{2}):(?P<min>\d{2}):(?P<s>\d{2})')


def _experiment_video_dataset_from_case_and_subcase(
    case_name: str, subcase_dir: Path
) -> LazyDescribed[ExperimentVideoDataset]:
    return LazyDescribed(
        LazyCallable(ExperimentVideoDataset)(
            _experiment_videos_from_subcase_dir(case_name, subcase_dir)
        ),
        f'{case_name}:{subcase_dir.name}',
    )


def _experiment_videos_from_subcase_dir(
    case_name: str, subcase_dir: Path
) -> Iterator[ExperimentVideo]:
    logger.info(f'Loading condensation dataset {case_name}:{subcase_dir.name}')

    for testdir in subcase_dir.iterdir():
        videopaths = (testdir / 'videos').glob('*.mp4')
        for video_path in videopaths:
            logger.debug(f'Adding video from {video_path}')
            video_name = video_path.stem
            ev_name = ':'.join((case_name, subcase_dir.name, testdir.name, video_name))
            yield ExperimentVideo(
                df_path=video_path.with_suffix('.csv'),
                video_path=video_path,
                name=ev_name,
            )


def _set_condensation_datasets_data(
    datasets: Iterable[LazyDescribed[ExperimentVideoDataset]], dataspecpath: PathLike
) -> tuple[LazyDescribed[ExperimentVideoDataset], ...]:
    logger.info('Setting condensation data')

    eager_datasets = tuple(dataset() for dataset in datasets)
    dataspecpath = resolve(dataspecpath)
    dataspec = yaml.safe_load(dataspecpath.read_text(encoding='utf8'))

    eager_datasets = tuple(_set_ev_dataset_data(dataset, dataspec) for dataset in eager_datasets)

    grouped_datasets = _group_datasets(eager_datasets)

    for dataset in grouped_datasets:
        _make_dataframe(dataset())

    logger.debug(f'Condensation datasets: {grouped_datasets}')

    return grouped_datasets


def _set_ev_dataset_data(
    dataset: ExperimentVideoDataset, dataspec: dict[str, Any]
) -> ExperimentVideoDataset:
    return ExperimentVideoDataset(
        ev for ev in (_set_ev_data(ev, dataspec) for ev in dataset) if ev is not None
    )


def _set_ev_data(ev: ExperimentVideo, dataspec: dict[str, Any]) -> ExperimentVideo | None:
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
        categories = {}

    categories['case'] = f'{case}:{subcase}'

    try:
        videospec = dataspec['cases'][case]['subcases'][subcase]['tests'][test_name]['videos'][
            f'{video_name}.mp4'
        ]
    except KeyError:
        # skip cases for which no data is available
        return None

    logger.debug(f'Getting video data for EV "{ev.name}"')

    videodata = VideoData(
        categories=categories,
        fps=Video(ev.path).fps(),
        # since there is no syncing between video and experimental data here,
        # we simply set the first frame as the reference
        ref_index=0,
        ref_elapsed_time=timedelta(0),
        start_elapsed_time=_parse_timedelta(videospec['start']),
        end_elapsed_time=_parse_timedelta(videospec['end']),
    )

    logger.debug(f'Setting video data for EV "{ev.name}"')
    return ev.with_data(videodata)


def _make_dataframe(dataset: ExperimentVideoDataset) -> None:
    missing: list[ExperimentVideo] = []
    for ev in dataset:
        try:
            logger.debug(f'Trying to load data for {ev.name}')
            ev.df = ev.load_df()
            logger.debug(f'Succesfully loaded data for {ev.name}')
        except FileNotFoundError:
            logger.debug(f'Failed to load data for {ev.name}')
            missing.append(ev)

    for ev in missing:
        logger.debug(f'Making dataframe for {ev.name}')

        ev.df = ev.make_dataframe(enforce_time=True)
        ev.save_df(ev.df)


def _group_datasets(
    datasets: Iterable[ExperimentVideoDataset],
) -> tuple[LazyDescribed[ExperimentVideoDataset], ...]:
    datasets_dict: dict[str, ExperimentVideoDataset] = {}
    for dataset in datasets:
        logger.debug(f'Reading condensation dataset {dataset}')

        for ev in dataset:
            if ev.data is None:
                # avoid including experiment videos that didn't have their data correctly set
                continue

            case, subcase, _test_name, _video_name = ev.name.split(':')
            dataset_name = f'{case}:{subcase}'

            if dataset_name not in datasets_dict:
                datasets_dict[dataset_name] = ExperimentVideoDataset()

            datasets_dict[dataset_name].add(ev)

    return tuple(LazyDescribed.from_describable(dataset) for dataset in datasets_dict.values())


def _parse_timedelta(s: Optional[str]) -> Optional[timedelta]:
    if s is None:
        return None

    m = _TIMEDELTA_PATTERN.fullmatch(s)

    if m is None:
        return None

    return timedelta(hours=int(m['h']), minutes=int(m['min']), seconds=int(m['s']))
