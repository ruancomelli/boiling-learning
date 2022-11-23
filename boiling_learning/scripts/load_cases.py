from typing import Iterable

from boiling_learning.preprocessing.cases import Case
from boiling_learning.utils.pathutils import PathLike


def main(
    casepaths: Iterable[PathLike], video_suffix: str, convert_videos: bool = False
) -> tuple[Case, ...]:
    cases = tuple(Case(casepath, video_suffix=video_suffix) for casepath in casepaths)

    if convert_videos:
        return tuple(case.convert_videos('.mp4', 'converted', overwrite=False) for case in cases)

    return cases
