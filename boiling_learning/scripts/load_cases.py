from typing import Iterable, Tuple

from boiling_learning.preprocessing.cases import Case
from boiling_learning.utils import PathLike, print_verbose


def main(
    casepaths: Iterable[PathLike],
    video_suffix: str,
    convert_videos: bool = False,
    verbose: bool = True,
) -> Tuple[Case, ...]:
    cases = tuple(Case(casepath, video_suffix=video_suffix) for casepath in casepaths)

    if convert_videos:
        for case in cases:
            print_verbose(verbose, f'Converting videos for case {case.name}')
            case.convert_videos('.mp4', 'converted', verbose=True, overwrite=False)

    return cases


if __name__ == '__main__':
    raise RuntimeError('*load_cases* cannot be executed as a standalone script yet.')
