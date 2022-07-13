from typing import Iterable, Tuple

from loguru import logger

from boiling_learning.preprocessing.cases import Case
from boiling_learning.utils.pathutils import PathLike


def main(
    casepaths: Iterable[PathLike], video_suffix: str, convert_videos: bool = False
) -> Tuple[Case, ...]:
    cases = tuple(Case(casepath, video_suffix=video_suffix) for casepath in casepaths)

    if convert_videos:
        for case in cases:
            logger.info(f'Converting videos for case {case.name}...')
            case.convert_videos('.mp4', 'converted', overwrite=False)
            logger.info(f'Successfully converted videos for case {case.name}...')

    return cases


if __name__ == '__main__':
    raise RuntimeError('*load_cases* cannot be executed as a standalone script yet.')
