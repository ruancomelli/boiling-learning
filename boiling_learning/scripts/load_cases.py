from typing import Iterable, Tuple

from dataclassy import dataclass

from boiling_learning.preprocessing import Case
from boiling_learning.utils import print_header, print_verbose
from boiling_learning.utils.pathutils import PathLike


@dataclass(frozen=True)
class Options:
    convert_videos: bool = False
    extract_audios: bool = False
    pre_load_videos: bool = False
    extract_frames: bool = False


def main(
    casepaths: Iterable[PathLike],
    video_suffix: str,
    options: Options,
    verbose: bool = True,
) -> Tuple[Case, ...]:
    cases = tuple(
        Case(casepath, video_suffix=video_suffix) for casepath in casepaths
    )

    for case in cases:
        if verbose:
            print_header(case.name)

        if options.convert_videos:
            print_verbose(verbose, 'Converting videos')
            case.convert_videos(
                '.mp4', 'converted', verbose=True, overwrite=False
            )
        if options.extract_audios:
            print_verbose(verbose, 'Extracting audios')
            case.extract_audios(verbose=True)
        if options.pre_load_videos:
            print_verbose(verbose, 'Opening videos')
            case.open_videos()
        if options.extract_frames:
            print_verbose(verbose, 'Extracting videos')
            case.extract_frames(
                overwrite=False,
                verbose=2,
                chunk_sizes=(100, 100),
                iterate=True,
            )

    return cases


if __name__ == '__main__':
    raise RuntimeError(
        '*interact_frames* cannot be executed as a standalone script yet.'
    )
