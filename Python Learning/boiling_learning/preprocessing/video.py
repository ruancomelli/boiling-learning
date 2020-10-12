import contextlib
import subprocess
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Union,
    Tuple
)
import string
import operator
import warnings

import funcy
import parse
from more_itertools import (
    ilen,
    peekable
)
import cv2
import numpy as np

import boiling_learning as bl
from boiling_learning.utils import (
    PathType,
    VerboseType
)
from boiling_learning.io.io import (
    make_callable_filename_pattern
)


def convert_video(
        in_path: PathType,
        out_path: PathType,
        remove_audio: bool = False,
        fps: Optional[Union[str, int, float]] = None,
        verbose: VerboseType = False,
        overwrite: bool = False,
) -> None:
    # For `fps`, see <https://superuser.com/a/729351>.

    in_path = bl.utils.ensure_resolved(in_path)
    out_path = bl.utils.ensure_parent(out_path)

    if verbose:
        print(
            'Converting video',
            bl.utils.shorten_path(in_path, max_len=50),
            '->',
            bl.utils.shorten_path(out_path, max_len=50)
        )

    if overwrite and out_path.is_file():
        if verbose:
            print('Overwriting', bl.utils.shorten_path(out_path, max_len=50))
        out_path.unlink()
        # TO-DO: in Python 3.8, use out_path.unlink(missing_ok=True) and remove one condition

    if out_path.is_file():
        if verbose:
            print(
                'Destination file already exists. Skipping video conversion.')
    else:
        command_list = [
            'ffmpeg',
            '-i', str(in_path),
            '-vsync', '0'
        ]
        if remove_audio:
            command_list.append('-an')
        if fps is not None:
            command_list.extend(['-r', str(fps)])
        command_list.append(str(out_path))

        if verbose:
            print(
                'Converting video:',
                bl.utils.shorten_path(in_path, max_len=40),
                '->',
                bl.utils.shorten_path(out_path, max_len=40)
            )
            print('Command list =', command_list)

        subprocess.run(command_list)


def extract_frames_ffmpeg(
        video_path: PathType,
        outputdir: PathType,
        filename_pattern: PathType = 'frame%d.png',
        overwrite: bool = False,
        fps: Optional[Union[str, int, float]] = None,
        verbose: VerboseType = False,
        # The image2 filter was used to process GOPRO images. Test if it's necessary
        image2filter: bool = False
) -> None:
    # Known ffmpeg commands to extract frames:
    # >>> ffmpeg -i {video_path} -r 1/1 -f image2 {output_path}
    # Source: <https://www.imore.com/how-extract-images-frame-frame-using-ffmpeg-macos>
    # >>> ffmpeg -i {video_path} {output_path}
    # >>> ffmpeg -i {video_path} -qscale:v 1 -vsync passthrough {output_path}
    # >>> ffmpeg -i {video_path} -qscale:v 1 -vsync 0 {output_path}
    # >>> ffmpeg -i {video_path} -f image2 {output_path}
    # See <https://stackoverflow.com/a/50422454/5811400>
    # >>> ffmpeg -i {video_path} -vsync 0 {output_path}

    video_path = bl.utils.ensure_resolved(video_path)
    outputdir = bl.utils.ensure_resolved(outputdir)

    if overwrite:
        bl.utils.rmdir(outputdir, recursive=True, missing_ok=True)

    output_path = bl.utils.ensure_parent(filename_pattern, root=outputdir)

    if verbose:
        print(
            'Extracting frames:',
            bl.utils.shorten_path(video_path, max_len=40),
            '->',
            bl.utils.shorten_path(output_path, max_len=40)
        )

    command_list = [
        'ffmpeg',
        '-i', str(video_path)
    ]
    if fps is not None:
        command_list.extend(['-r', str(fps)])
    if image2filter:
        command_list.extend(['-f', 'image2'])
    command_list.append(str(output_path))

    if verbose:
        print('Command list =', command_list)

    subprocess.run(command_list)


def make_callable_index_parser(
        index_parser: Union[PathType, Callable[[PathType], int]],
        index_key: Optional[str] = None
) -> Tuple[bool, Callable[[PathType], int]]:

    if callable(index_parser):
        return True, index_parser
    else:
        index_parser_str = str(index_parser)

        if index_key is not None and index_key in {
                tup[1]
                for tup in string.Formatter().parse(index_parser_str)
                if tup[1] is not None
        }:
            parser = parse.compile(index_parser_str).parse
            index_parser = funcy.compose(
                int,
                operator.itemgetter(index_key),
                parser,
                str
            )
            return True, index_parser
        else:
            return False, index_parser
            # raise ValueError('index_parser must be either a callable or a str-convertible value. The resulting str must contain index_key as a key for parsing.')

            # try:
            #     index_parser_str % 0  # checks if it is possible to use old-style formatting
            # except TypeError:
            #     return False, index_parser

            # def _filename_pattern(index: int) -> Path:
            #     return bl.utils.ensure_parent(
            #         index_parser_str % index,
            #         root=inputdir
            #     )

            # return True, _filename_pattern


def extract_frames_iterate(
        video_path: PathType,
        outputdir: PathType,
        filename_pattern: Union[PathType, Callable[[int], PathType]],
        overwrite: bool,
        index_key: Optional[str] = None,
        verbose: VerboseType = False
) -> None:
    verbose_2 = verbose >= 2
    video_path = bl.utils.ensure_resolved(video_path)

    if video_path.suffix != '.mp4':
        warnings.warn(
            'Video suffix is not .mp4'
            '- this may be troublesome as other formats'
            'are known to iterate incorrectly.'
            'Consider converting your video to .mp4 first.',
            category=warnings.RuntimeWarning
        )

    success, filename_pattern = make_callable_filename_pattern(
        outputdir,
        filename_pattern,
        index_key=index_key
    )
    if not success:
        raise ValueError(
            'filename_pattern could not be successfully converted to a callable.')

    if verbose:
        print('Extracting frames iteratively.')

    for index, frame in enumerate(frames(video_path)):
        path = filename_pattern(index)

        if verbose_2:
            print(
                f'Frame #{index}',
                '->',
                bl.utils.shorten_path(path, max_len=60),
                '...',
                end=' '
            )

        if not overwrite and path.is_file():
            if verbose_2:
                print('skip.')
            continue
        if verbose_2:
            print('write.')

        cv2.imwrite(str(path), frame)


def extracted_frames_count(
        video_path: PathType,
        outputdir: PathType,
        frame_suffix: str,
        tmp_dir: Optional[PathType] = None,
        fast_frames_count: bool = False,
        metadata_path: Optional[PathType] = None,
        recount_source: bool = False,
        recount_tmp: bool = False,
        recount_dest: bool = False,
        verbose: VerboseType = False
) -> Tuple[int, Optional[int], int]:
    use_metadata = metadata_path is not None
    use_tmp_dir = tmp_dir is not None
    fast_key = 'fast' if fast_frames_count else 'slow'

    video_path = bl.utils.ensure_resolved(video_path)
    outputdir = bl.utils.ensure_resolved(outputdir)

    if verbose:
        print('Counting frames from file', video_path)
        print('Counting frames from directory', outputdir)

    video_frames_count = None
    if use_metadata:
        metadata_path = bl.utils.ensure_parent(metadata_path, root=outputdir)
        if metadata_path.is_file():
            metadata = bl.io.load_json(metadata_path)
        else:
            metadata = dict()

    if use_metadata and not recount_source:
        video_frames_count = metadata.get('video', {}).get(fast_key)

    if video_frames_count is None:
        video_frames_count = count_frames(video_path, fast=fast_frames_count)

    if use_metadata:
        metadata.setdefault('video', {})[fast_key] = video_frames_count
        bl.io.save_json(metadata, metadata_path)

    tmp_dir_count = None
    if use_tmp_dir:
        rel_tmp_dir = bl.utils.relative_path(outputdir, tmp_dir)
        if use_metadata and not recount_tmp:
            tmp_dir_count = metadata.get('tmp_dir', {}).get(rel_tmp_dir)
        if tmp_dir_count is None:
            tmp_dir_count = count_frames_in_dir(tmp_dir)
        if use_metadata:
            metadata.setdefault('tmp_dir', {})[rel_tmp_dir] = tmp_dir_count
            bl.io.save_json(metadata, metadata_path)

    extracted_count = None
    if use_metadata and not recount_dest:
        extracted_count = metadata.get('extracted')

    if extracted_count is None:
        if use_tmp_dir:
            extracted_count = count_frames_in_dir(
                outputdir, frame_suffix=frame_suffix,
                exclude=tmp_dir, exclude_count=tmp_dir_count)
        else:
            extracted_count = count_frames_in_dir(
                outputdir, frame_suffix=frame_suffix)

    if use_metadata:
        metadata['extracted'] = extracted_count
        bl.io.save_json(metadata, metadata_path)

    if verbose:
        print('Video frames count:', video_frames_count)
        if use_tmp_dir:
            print('Frames count in temporary folder:', tmp_dir_count)
        print('Extracted frames count:', extracted_count)

    return video_frames_count, tmp_dir_count, extracted_count


def extract_frames(
    video_path: PathType,
    outputdir: PathType,
    filename_pattern: Union[PathType, Callable[[int], PathType]] = 'frame%d.png',
    index_key: Optional[str] = None,
    frame_suffix: Optional[str] = None,
    verbose: VerboseType = False,
    fast_frames_count: Optional[bool] = None,
    iterate: bool = False,
    overwrite: bool = False,
    tmp_dir: Optional[PathType] = None,
    metadata_path: Optional[PathType] = None
) -> None:
    # Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
    # Source 2: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>
    # TODO: improve error messages

    video_path = bl.utils.ensure_resolved(video_path)
    outputdir = bl.utils.ensure_dir(outputdir)
    if verbose:
        print(
            'Extracting frames:',
            bl.utils.shorten_path(video_path, max_len=52),
            '->',
            bl.utils.shorten_path(outputdir, max_len=52)
        )
    verbose_2 = verbose >= 2

    # Check for invalid input
    use_frames_count = fast_frames_count is not None
    if use_frames_count and overwrite:
        raise ValueError('cannot overwrite when frames count mode is passed.')

    use_persistent_tmp_dir = tmp_dir is not None
    if use_persistent_tmp_dir:
        if iterate:
            raise ValueError('cannot use tmp_dir with iterative extraction.')
        else:
            tmp_dir = bl.utils.ensure_dir(tmp_dir, root=outputdir)

            def rm_tmp_dir() -> None:
                if verbose:
                    print('Removing temporary folder.')
                bl.utils.rmdir(tmp_dir, recursive=True, missing_ok=True)

            if verbose:
                print('Using persistent temporary folder at', tmp_dir)

    use_tmp_dir = not callable(filename_pattern)
    if frame_suffix is None:
        if use_tmp_dir:
            frame_suffix = Path(filename_pattern).suffix
        elif use_frames_count:
            raise ValueError(
                'when filename_pattern is callable'
                'and a frames count mode is used,'
                'frames suffixes must be explicitly given as argument.'
            )

    callable_filename_pattern, callable_filename_pattern = make_callable_filename_pattern(
        outputdir, filename_pattern, index_key
    )

    if use_frames_count:
        video_frames_count, tmp_dir_count, extracted_count = extracted_frames_count(
            video_path,
            outputdir,
            frame_suffix,
            tmp_dir,
            fast_frames_count,
            metadata_path,
            verbose
        )

    skip_tmp_extraction = use_frames_count and tmp_dir_count is not None and tmp_dir_count == video_frames_count
    skip_extraction = use_frames_count and extracted_count == video_frames_count

    if skip_extraction:
        if verbose:
            print('Frames already extracted. Skipping.')
        return

    if iterate:
        extract_frames_iterate(
            video_path=video_path,
            outputdir=outputdir,
            filename_pattern=callable_filename_pattern,
            overwrite=overwrite,
            index_key=index_key,
            verbose=verbose
        )
    elif use_tmp_dir:
        if use_persistent_tmp_dir:
            cm = bl.utils.nullcontext(tmp_dir)
        else:
            cm = bl.utils.tempdir(prefix='_', dir=outputdir)

        with cm as temporary_folder:
            if verbose:
                print('Extracting frames to temporary folder',
                      bl.utils.shorten_path(temporary_folder, max_len=40))

            tmp_format = f'frame%d{frame_suffix}'
            tmp_parser = f'frame{{index:d}}{frame_suffix}'

            if skip_tmp_extraction:
                if verbose:
                    print('Skipping frames extraction to temporary folder.')
            else:
                extract_frames_ffmpeg(
                    video_path,
                    temporary_folder,
                    filename_pattern=tmp_format,
                    overwrite=True,
                    verbose=verbose
                )

            source_dest_pairs = (
                (source, callable_filename_pattern(
                    parse.parse(tmp_parser, source.name)['index']))
                for source in temporary_folder.iterdir()
            )

            source_dest_pairs = filter(
                lambda source_dest_pair: source_dest_pair[0].is_file(),
                source_dest_pairs)

            if not overwrite:
                source_dest_pairs = filter(
                    lambda source_dest_pair: not source_dest_pair[1].is_file(),
                    source_dest_pairs)

            source_dest_pairs = peekable(source_dest_pairs)
            if source_dest_pairs:  # if there are pairs to transform
                for source, dest in source_dest_pairs:
                    if verbose_2:
                        print('Moving', bl.utils.shorten_path(source, 30),
                              'to', bl.utils.shorten_path(dest, 30))
                    dest = bl.utils.ensure_parent(dest)
                    source.rename(dest)
            elif use_persistent_tmp_dir:
                rm_tmp_dir()
    else:
        extract_frames_ffmpeg(
            video_path,
            outputdir,
            filename_pattern=callable_filename_pattern,
            overwrite=overwrite,
            verbose=verbose
        )


def concat_videos(
        in_paths: Iterable[PathType],
        out_path: PathType
) -> None:
    # Original command: ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
    # Source: <https://stackoverflow.com/a/11175851/5811400>
    # TODO: test!

    in_paths = map(bl.utils.ensure_resolved, in_paths)
    out_path = bl.utils.ensure_parent(out_path)
    out_dir = out_path.parent

    with bl.utils.tempdir(prefix='_', dir=out_dir) as temp_dir:
        input_file_path = temp_dir / 'input.txt'

        with open(input_file_path, 'w+') as fp:
            fp.write(
                '\n'.join(
                    f'file {in_path}'
                    for in_path in in_paths
                )
            )

        command_list = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(input_file_path),
            '-c', 'copy',
            str(out_path)
        ]
        subprocess.run(command_list)


# Original code: $ ffmpeg -i input.mp4 -c:a copy -vn -sn output.m4a
# Source: <https://superuser.com/a/633765>
def extract_audio(
        video_path: PathType,
        out_path: PathType,
        overwrite: bool = False,
        verbose: bool = False
) -> None:
    out_path = bl.utils.ensure_parent(out_path)

    if verbose:
        print(
            'Extracting audio:',
            bl.utils.shorten_path(video_path, max_len=52),
            '->',
            bl.utils.shorten_path(out_path, max_len=52)
        )

    if overwrite or not out_path.is_file():
        if verbose:
            print(
                'Audio does not exist or overwrite mode is on. Extracting...',
                end=' ')

        command_list = [
            'ffmpeg',
            '-i', str(video_path),
            '-c:a', 'copy',
            '-vn',
            '-sn',
            str(out_path)
        ]
        subprocess.run(command_list)

        bl.utils.print_verbose(verbose, f'Done.')
    elif verbose:
        print(f'Audio already exists. Skipping proccess.')


@contextlib.contextmanager
def open_video(video_path: PathType) -> Iterator[cv2.VideoCapture]:
    video_path = bl.utils.ensure_resolved(video_path)
    cap = cv2.VideoCapture(str(video_path))

    try:
        yield cap
    finally:
        cap.release()


def frames(
        video_path: PathType,
        suppress_retrieval_failure: bool = True
) -> Iterator[np.ndarray]:
    # Does not work with GOPRO format

    with open_video(video_path) as cap:
        while cap.grab():
            flag, frame = cap.retrieve()
            if flag:
                yield frame
            elif not suppress_retrieval_failure:
                raise RuntimeError(
                    f'failed frame retrieval for video at {video_path}')


def count_frames(video_path: PathType, fast: bool = False) -> int:
    if fast:
        with open_video(video_path) as cap:
            return int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    else:
        return ilen(frames(video_path))


def count_frames_in_dir(
        path: PathType,
        frame_suffix: str,
        recursive: bool = True,
        exclude_path: Optional[PathType] = None,
        exclude_count: Optional[Union[int, Callable[[PathType], int]]] = None
) -> int:
    path = bl.utils.ensure_resolved(path)
    if not path.is_dir():
        return 0

    if recursive:
        globber = path.rglob
    else:
        globber = path.glob

    n_frames_in_path = ilen(globber(f'*{frame_suffix}'))
    if exclude_path is None:
        return n_frames_in_path
    else:
        exclude_path = bl.utils.ensure_resolved(exclude_path)
        if bl.utils.is_parent_dir(path, exclude_path):
            if exclude_count is None:
                exclude_count = count_frames_in_dir(
                    exclude_path, frame_suffix, recursive=recursive)
            elif callable(exclude_count):
                exclude_count = exclude_count(exclude_path)

            return n_frames_in_path - exclude_count
        else:
            return n_frames_in_path


def reorganize_frames(
        dest_dir: PathType,
        filename_pattern: Union[PathType, Callable[[int], PathType]],
        index_parser: Union[PathType, Callable[[str], int]],
        source_dir: Optional[PathType] = None,
        source_files: Optional[Iterable[PathType]] = None,
        index_key: Optional[str] = None,
        source_suffix: str = '',
        overwrite: bool = False,
        verbose: VerboseType = False
) -> None:
    # TODO: check for incompatible arguments

    if source_dir is None and source_files is None:
        raise ValueError(
            'Either source_dir is a PathType or source_files is an iterable yielding PathType')

    if source_suffix and not source_suffix.startswith('.'):
        raise ValueError(
            'source_suffix must either be the empty string \'\' or start with a dot \'.\'')

    success, filename_pattern = make_callable_filename_pattern(
        dest_dir, filename_pattern, index_key
    )

    if not success:
        raise ValueError(
            'filename_pattern could not be converted to a callable.')

    success, index_parser = make_callable_index_parser(
        index_parser, index_key
    )

    if not success:
        raise ValueError('index_parser could not be converted to a callable.')

    if source_files is None:
        source_dir = bl.utils.ensure_resolved(source_dir)
        source_files = source_dir.rglob('*' + source_suffix)

    indices = map(index_parser, source_files)
    dest_files = map(filename_pattern, indices)

    if overwrite:
        src_dest_pairs = zip(source_files, dest_files)
    else:
        src_dest_pairs = bl.utils.functional.zip_filter(
            lambda src, dst: not dst.is_file(),
            source_files,
            dest_files)

    if verbose >= 2:
        def renamer(src, dest):
            print(' -> '.join([
                bl.utils.shorten_path(src, max_len=60),
                bl.utils.shorten_path(dest, max_len=60)
            ]))
            src.rename(dest)
    else:
        def renamer(src, dest):
            src.rename(dest)

    bl.utils.functional.starapply(renamer, src_dest_pairs)
