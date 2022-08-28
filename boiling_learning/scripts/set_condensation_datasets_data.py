import re
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import parse
import yaml
from loguru import logger

from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.preprocessing.video import Video
from boiling_learning.scripts.utils.setting_data import check_experiment_video_dataframe_indices
from boiling_learning.utils.frozendicts import frozendict
from boiling_learning.utils.pathutils import PathLike, resolve

_SUBCASE_PATTERNS = frozendict(
    {
        'T_inf': parse.compile('T_inf {:d}C'),
        'T_s': parse.compile('T_s {:d}C'),
        'rh': parse.compile('rh {:d}%'),
    }
)
_TIMEDELTA_PATTERN = re.compile(r'(?P<h>\d{2}):(?P<min>\d{2}):(?P<s>\d{2})')


def main(datasets: Iterable[ImageDataset], dataspecpath: PathLike) -> Tuple[ImageDataset, ...]:
    logger.info('Setting condensation data')

    datasets = tuple(datasets)
    dataspecpath = resolve(dataspecpath)
    dataspec = yaml.safe_load(dataspecpath.read_text())

    for dataset in datasets:
        logger.debug(f'Reading condensation dataset {dataset.name}')

        for ev in dataset:
            _set_ev_data(ev, dataspec)

    grouped_datasets = _group_datasets(datasets)

    for dataset in grouped_datasets:
        _make_dataframe(dataset)

    logger.debug(f'Condensation datasets: {grouped_datasets}')

    return grouped_datasets


def _set_ev_data(ev: ExperimentVideo, dataspec: Dict[str, Any]) -> None:
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
        return

    logger.debug(f'Getting video data for EV "{ev.name}"')

    videodata = ExperimentVideo.VideoData(
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

        ev.make_dataframe(exist_load=True, enforce_time=True, inplace=True)

        check_experiment_video_dataframe_indices(ev)

        ev.save_df(overwrite=False)


def _group_datasets(datasets: Iterable[ImageDataset]) -> Tuple[ImageDataset, ...]:
    datasets_dict: Dict[str, ImageDataset] = {}
    for dataset in datasets:
        logger.debug(f'Reading condensation dataset {dataset.name}')

        for ev in dataset:
            if ev.data is None:
                # avoid including experiment videos that didn't have their data correctly set
                continue

            case, subcase, _test_name, _video_name = ev.name.split(':')
            dataset_name = f'{case}:{subcase}'

            if dataset_name not in datasets_dict:
                datasets_dict[dataset_name] = ImageDataset(dataset_name)

            datasets_dict[dataset_name].add(ev)

    return tuple(datasets_dict.values())


if __name__ == '__main__':
    raise RuntimeError(
        '*set_condensation_datasets_data* cannot be executed as a standalone script yet.'
    )
