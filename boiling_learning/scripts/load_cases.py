from typing import Iterable, Tuple

from dataclassy import dataclass

from boiling_learning.preprocessing.cases import Case
from boiling_learning.utils import PathLike, print_header, print_verbose


@dataclass(frozen=True)
class Options:
    convert_videos: bool = False
    pre_load_videos: bool = False


def main(
    casepaths: Iterable[PathLike],
    video_suffix: str,
    options: Options,
    verbose: bool = True,
) -> Tuple[Case, ...]:
    cases = tuple(Case(casepath, video_suffix=video_suffix) for casepath in casepaths)

    for case in cases:
        if verbose:
            print_header(case.name)

        if options.convert_videos:
            print_verbose(verbose, 'Converting videos')
            case.convert_videos('.mp4', 'converted', verbose=True, overwrite=False)
        if options.pre_load_videos:
            print_verbose(verbose, 'Opening videos')
            case.open_videos()

    return cases


if __name__ == '__main__':
    raise RuntimeError('*load_cases* cannot be executed as a standalone script yet.')
