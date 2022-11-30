from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from boiling_learning.app.paths import DATA_PATH
from boiling_learning.lazy import LazyCallable, LazyDescribed
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset

CONDENSATION_DATA_PATH = DATA_PATH / 'condensation'
CONDENSATION_DATA_SPEC_PATH = CONDENSATION_DATA_PATH / 'data_spec.yaml'


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


CONDENSATION_DATASETS = tuple(
    _experiment_video_dataset_from_case_and_subcase(case_dir.name, subcase_dir)
    for case_dir in CONDENSATION_DATA_PATH.iterdir()
    if case_dir.is_dir()
    for subcase_dir in case_dir.iterdir()
    if subcase_dir.is_dir()
)
