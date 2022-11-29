from typing import Iterable

from loguru import logger

from boiling_learning.app.paths import DATA_PATH
from boiling_learning.preprocessing.cases import Case
from boiling_learning.utils.pathutils import PathLike

BOILING_DATA_PATH = DATA_PATH / 'boiling1d'
BOILING_EXPERIMENTS_PATH = BOILING_DATA_PATH / 'experiments'


def load_experiment_video_datasets() -> tuple[Case, ...]:
    logger.info(f'Loading boiling cases from {BOILING_DATA_PATH}')
    return _load_experiment_video_datasets(
        (BOILING_DATA_PATH / case_name for case_name in ('case 1', 'case 2', 'case 3', 'case 4')),
        video_suffix='.MP4',
    )


def _load_experiment_video_datasets(
    casepaths: Iterable[PathLike], video_suffix: str
) -> tuple[Case, ...]:
    return tuple(
        Case(casepath, video_suffix=video_suffix).convert_videos(
            '.mp4', 'converted', overwrite=False
        )
        for casepath in casepaths
    )
