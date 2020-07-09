import functools
import contextlib
import subprocess
import os
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

from parse import parse
from more_itertools import (
    ilen,
    unzip,
    peekable,
    consume
)
import cv2
import numpy as np

import boiling_learning as bl
from boiling_learning.utils import PathType

# Known ffmpeg commands to extract frames:
# >>> ffmpeg -i {video_path} -r 1/1 -f image2 {output_path}
# Source: <https://www.imore.com/how-extract-images-frame-frame-using-ffmpeg-macos>
# >>> ffmpeg -i {video_path} {output_path}
# >>> ffmpeg -i {video_path} -qscale:v 1 -vsync passthrough {output_path}
# >>> ffmpeg -i {video_path} -qscale:v 1 -vsync 0 {output_path}
# >>> ffmpeg -i {video_path} -f image2 {output_path}
# See <https://stackoverflow.com/a/50422454/5811400>
# >>> ffmpeg -i {video_path} -vsync 0 {output_path}


def convert_video(
        in_path: PathType,
        out_path: PathType,
        remove_audio: bool = False,
        verbose: Union[bool, int] = False,
        overwrite: bool = False,
) -> None:
    in_path = bl.utils.ensure_resolved(in_path)
    out_path = bl.utils.ensure_parent(out_path)

    if overwrite and out_path.is_file():
        out_path.unlink()

    if not out_path.is_file():
        command_list = [
            'ffmpeg',
            '-i',
            str(in_path),
            '-vsync',
            '0'
        ]
        if remove_audio:
            command_list += ['-an']
        command_list += [
            str(out_path)
        ]

        if verbose:
            print(
                f'Converting video: "{bl.utils.shorten_path(in_path, max_len=40)}" -> "{bl.utils.shorten_path(out_path, max_len=40)}"')
            print('Command list =', command_list)

        subprocess.run(command_list)


def extract_frames_ffmpeg(
        video_path: PathType,
        outputdir: PathType,
        filename_pattern: PathType = 'frame%d.png',
        overwrite: bool = False,
        verbose: Union[bool, int] = False,
        # The image2 filter was used to process GOPRO images. Test if it's necessary
        image2filter: bool = False
) -> None:
    video_path = bl.utils.ensure_resolved(video_path)
    outputdir = bl.utils.ensure_resolved(outputdir)

    if overwrite:
        bl.utils.rmdir(outputdir, recursive=True, missing_ok=True)

    output_path = bl.utils.ensure_parent(filename_pattern, root=outputdir)

    if verbose:
        print(
            f'Extracting frames: "{bl.utils.shorten_path(video_path, max_len=40)}" -> "{bl.utils.shorten_path(output_path, max_len=40)}"')

    if image2filter:
        command_list = [
            'ffmpeg',
            '-i',
            f'{video_path}',
            '-f',
            'image2',
            f'{output_path}'
        ]
    else:
        command_list = [
            'ffmpeg',
            '-i',
            f'{video_path}',
            f'{output_path}'
        ]

    if verbose:
        print(f'Command list = {command_list}')

    subprocess.run(command_list)


def make_callable_filename_pattern(
        outputdir: PathType,
        filename_pattern: Union[PathType, Callable[[int], PathType]],
        index_key: Optional[str] = None
) -> Tuple[bool, Callable[[int], Path]]:

    if callable(filename_pattern):
        def _filename_pattern(index: int) -> Path:
            return bl.utils.ensure_parent(
                filename_pattern(index),
                root=outputdir
            )

        return True, _filename_pattern
    else:
        filename_pattern_str = str(filename_pattern)

        if index_key is not None and index_key in {
                tup[1]
                for tup in string.Formatter().parse(filename_pattern_str)
                if tup[1] is not None
        }:
            formatter = filename_pattern_str.format

            def _filename_pattern(index: int) -> Path:
                return bl.utils.ensure_parent(
                    formatter(
                        **{index_key: index}
                    ),
                    root=outputdir
                )
            return True, _filename_pattern

        else:
            try:
                filename_pattern_str % 0  # checks if it is possible to use old-style formatting
            except TypeError:
                return False, filename_pattern

            def _filename_pattern(index: int) -> Path:
                return bl.utils.ensure_parent(
                    filename_pattern_str % index,
                    root=outputdir
                )

            return True, _filename_pattern


def extract_frames_iterate(
        video_path: PathType,
        outputdir: PathType,
        filename_pattern: Union[PathType, Callable[[int], PathType]],
        overwrite: bool,
        index_key: Optional[str] = None,
        verbose: Union[bool, int] = False
) -> None:
    verbose_2 = verbose >= 2

    video_path = bl.utils.ensure_resolved(video_path)
    success, filename_pattern = make_callable_filename_pattern(
        outputdir,
        filename_pattern,
        index_key=index_key
    )
    if not success:
        raise ValueError(
            'filename_pattern could not be successfully converted to a callable.')

    bl.utils.print_verbose(verbose, 'Extracting frames iteratively.')

    for index, frame in enumerate(frames(video_path)):
        path = filename_pattern(index)

        if verbose_2:
            print(
                f'Frame #{index} -> {bl.utils.shorten_path(path, max_len=60)} ... ', end='')

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
        verbose: bool = False
) -> Tuple[int, Optional[int], int]:
    use_metadata = metadata_path is not None
    use_tmp_dir = tmp_dir is not None
    fast_key = 'fast' if fast_frames_count else 'slow'

    video_frames_count = None
    if use_metadata:
        metadata_path = bl.utils.ensure_parent(metadata_path, root=outputdir)
        if metadata_path.is_file():
            metadata = bl.io.load_json(metadata_path)
            video_frames_count = metadata.get('video', {}).get(fast_key)
        else:
            metadata = dict()

    if video_frames_count is None:
        video_frames_count = count_frames(video_path, fast=fast_frames_count)

    if use_metadata:
        metadata.setdefault('video', {})[fast_key] = video_frames_count
        bl.io.save_json(metadata, metadata_path)

    tmp_dir_count = None
    if use_tmp_dir:
        rel_tmp_dir = bl.utils.relative_path(outputdir, tmp_dir)
        if use_metadata:
            tmp_dir_count = metadata.get('tmp_dir', {}).get(rel_tmp_dir)
        if tmp_dir_count is None:
            tmp_dir_count = count_frames_in_dir(tmp_dir)
        if use_metadata:
            metadata.setdefault('tmp_dir', {})[rel_tmp_dir] = tmp_dir_count
            bl.io.save_json(metadata, metadata_path)

        extracted_count = count_frames_in_dir(
            outputdir, frame_suffix=frame_suffix, exclude=tmp_dir, exclude_count=tmp_dir_count)
    else:
        extracted_count = count_frames_in_dir(
            outputdir, frame_suffix=frame_suffix)

    if use_metadata:
        metadata['extracted'] = tmp_dir_count
        bl.io.save_json(metadata, metadata_path)

    bl.utils.print_verbose(
        verbose, f'Video frames count: {video_frames_count}')
    if use_tmp_dir:
        bl.utils.print_verbose(
            verbose, f'Frames count in temporary folder: {tmp_dir_count}')
    bl.utils.print_verbose(
        verbose, f'Extracted frames count: {extracted_count}')

    return video_frames_count, tmp_dir_count, extracted_count


def extract_frames(
    video_path: PathType,
    outputdir: PathType,
    filename_pattern: Union[PathType, Callable[[int], PathType]] = 'frame%d.png',
    index_key: Optional[str] = None,
    frame_suffix: Optional[str] = None,
    verbose: Union[bool, int] = False,
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
            f'Extracting: {bl.utils.shorten_path(video_path, max_len=52)} -> {bl.utils.shorten_path(outputdir, max_len=52)}')
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
                bl.utils.print_verbose(verbose, 'Removing temporary folder.')
                bl.utils.rmdir(tmp_dir, recursive=True, missing_ok=True)

            bl.utils.print_verbose(
                verbose, f'Using persistent temporary folder at {tmp_dir}')

    use_tmp_dir = not callable(filename_pattern)
    if frame_suffix is None:
        if use_tmp_dir:
            frame_suffix = Path(filename_pattern).suffix
        elif use_frames_count:
            raise ValueError(
                'when filename_pattern is callable and a frames count mode is used, frames suffixes must be explicitly given as argument.'
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
        bl.utils.print_verbose(verbose, f'Frames already extracted. Skipping.')
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
                bl.utils.print_verbose(
                    verbose, 'Skipping frames extraction to temporary folder.')
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
                    parse(tmp_parser, source.name)['index']))
                for source in temporary_folder.iterdir()
            )

            source_dest_pairs = filter(
                lambda source_dest_pair: source_dest_pair[0].is_file(), source_dest_pairs)

            if not overwrite:
                source_dest_pairs = filter(
                    lambda source_dest_pair: not source_dest_pair[1].is_file(), source_dest_pairs)

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
        input_file = temp_dir / 'input.txt'

        with open(input_file, 'w+') as fp:
            fp.write(
                '\n'.join(
                    f'file {in_path}' for in_path in in_paths
                )
            )

        command_list = [
            'ffmpeg',
            '-f',
            'concat',
            '-safe',
            '0',
            '-i',
            f'{input_file}',
            '-c',
            'copy',
            f'{out_path}'
        ]
        subprocess.run(command_list)


# Original code: $ ffmpeg -i input.mp4 -c:a copy -vn -sn output.m4a
# Source: <https://superuser.com/a/633765>
def extract_audio(
        video_path: PathType,
        out_path: PathType
) -> None:
    out_path = bl.utils.ensure_parent(out_path)

    command_list = [
        'ffmpeg',
        '-i',
        f'{video_path}',
        '-c:a',
        'copy',
        '-vn',
        '-sn',
        f'{out_path}'
    ]
    subprocess.run(command_list)


@contextlib.contextmanager
def open_video(video_path: PathType) -> Iterator[cv2.VideoCapture]:
    video_path = bl.utils.ensure_resolved(video_path)
    cap = cv2.VideoCapture(str(video_path))

    try:
        yield cap
    finally:
        cap.release()


def frames(video_path: PathType, suppress_retrieval_failure: bool = True) -> Iterator[np.ndarray]:
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
