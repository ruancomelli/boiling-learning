from pathlib import Path

from loguru import logger

from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.utils.pathutils import PathLike, resolve


def main(data_path: PathLike, /) -> list[ExperimentVideoDataset]:
    logger.info(f'Loading cases from {data_path}')

    data_path = resolve(data_path)

    return [
        _experiment_video_dataset_from_case_and_subcase(case_dir.name, subcase_dir)
        for case_dir in data_path.iterdir()
        if case_dir.is_dir()
        for subcase_dir in case_dir.iterdir()
        if subcase_dir.is_dir()
    ]


def _experiment_video_dataset_from_case_and_subcase(
    case_name: str, subcase_dir: Path
) -> ExperimentVideoDataset:
    dataset = ExperimentVideoDataset()
    subcase_name = subcase_dir.name
    for testdir in subcase_dir.iterdir():
        logger.debug(f'Searching for videos in {testdir}')

        test_name = testdir.name

        videopaths = (testdir / 'videos').glob('*.mp4')
        for video_path in videopaths:
            logger.debug(f'Adding video from {video_path}')
            video_name = video_path.stem
            ev_name = ':'.join((case_name, subcase_name, test_name, video_name))
            ev = ExperimentVideo(
                df_path=video_path.with_suffix('.csv'),
                video_path=video_path,
                name=ev_name,
            )
            dataset.add(ev)
    return dataset
