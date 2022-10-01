from typing import List

from loguru import logger

from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.utils.pathutils import PathLike, resolve


def main(datapath: PathLike) -> List[ExperimentVideoDataset]:
    logger.info(f'Loading cases from {datapath}')

    datapath = resolve(datapath)

    datasets: List[ExperimentVideoDataset] = []
    for casedir in datapath.iterdir():
        logger.debug(f'Searching for subcases in {casedir}')
        if not casedir.is_dir():
            continue

        case = casedir.name
        for subcasedir in casedir.iterdir():
            logger.debug(f'Searching for tests in {subcasedir}')

            if not subcasedir.is_dir():
                continue

            subcase = subcasedir.name

            dataset = ExperimentVideoDataset(f'{case}:{subcase}')
            for testdir in subcasedir.iterdir():
                logger.debug(f'Searching for videos in {testdir}')

                test_name = testdir.name

                videopaths = (testdir / 'videos').glob('*.mp4')
                for video_path in videopaths:
                    logger.debug(f'Adding video from {video_path}')
                    video_name = video_path.stem
                    ev_name = ':'.join((case, subcase, test_name, video_name))
                    ev = ExperimentVideo(
                        df_path=video_path.with_suffix('.csv'),
                        video_path=video_path,
                        name=ev_name,
                    )
                    dataset.add(ev)

            datasets.append(dataset)
    return datasets


if __name__ == '__main__':
    raise RuntimeError('*load_dataset_tree* cannot be executed as a standalone script yet.')
