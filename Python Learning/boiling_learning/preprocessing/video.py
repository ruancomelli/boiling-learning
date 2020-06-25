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
    Union
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
    metapath: Optional[PathType] = None
) -> None:
    # Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
    # Source 2: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>

    verbose_2 = int(verbose) >= 2

    use_frames_count = fast_frames_count is not None
    if use_frames_count and overwrite:
        raise ValueError('cannot overwrite when frames count mode is passed.')

    use_persistent_tmp_dir = tmp_dir is not None
    if use_persistent_tmp_dir and iterate:
        raise ValueError('cannot use tmp_dir with iterative extraction.')

    callable_filename_pattern = callable(filename_pattern)
    if (
            callable_filename_pattern
            and frame_suffix is None
            and use_frames_count
    ):
        raise ValueError(
            'when filename_pattern is callable and a frames count mode is used, frames suffixes must be explicitly given as argument.'
        )

    video_path = bl.utils.ensure_resolved(video_path)
    outputdir = bl.utils.ensure_resolved(outputdir)
    bl.utils.print_verbose(
        verbose, f'Extracting: {bl.utils.shorten_path(video_path, max_len=52)} -> {bl.utils.shorten_path(outputdir, max_len=52)}')
    
    use_metadata = metapath is not None
    if use_metadata:
        metapath = bl.utils.ensure_parent(metapath, root=outputdir)
        if not metapath.is_file():
            bl.io.save_json({}, metapath)
        metadata = bl.io.load_json(metapath)
        
        video_frames_key = 'video_frames_count'
        fast_key = 'fast' if fast_frames_count else 'slow'
        # extracted_frames_key = 'extracted_frames_count'
        # tmp_dir_frames_key = 'temporary_dir_frames_count'

    if not callable_filename_pattern:
        # try:
        #     filename_pattern_str = filename_pattern.__fspath__()
        # except AttributeError:
            # filename_pattern_str = str(filename_pattern)
        filename_pattern_str = str(filename_pattern)

        if frame_suffix is None:
            frame_suffix = Path(filename_pattern_str).suffix

        # Source: <https://stackoverflow.com/a/46161774/5811400>
        if index_key in {
                tup[1]
                for tup in string.Formatter().parse(filename_pattern_str)
                if tup[1] is not None
        }:
            formatter = filename_pattern.format

            def filename_pattern(index: int) -> str:
                return formatter(
                    **{index_key: index}
                )

            callable_filename_pattern = True

    if callable_filename_pattern:
        filename_pattern = bl.utils.functional.compose(
            filename_pattern,
            functools.partial(bl.utils.ensure_resolved, root=outputdir)
        )  # make sure that filename_pattern outputs absolute paths

    if use_persistent_tmp_dir:
        tmp_dir = bl.utils.ensure_resolved(tmp_dir, root=outputdir)

        def rm_tmp_dir() -> None:
            bl.utils.print_verbose(verbose, 'Removing temporary folder.')
            bl.utils.rmdir(tmp_dir, recursive=True, missing_ok=True)

        bl.utils.print_verbose(
            verbose, f'Using persistent temporary folder at {tmp_dir}')
        
        if use_frames_count:
            tmp_dir_count = count_frames_in_dir(tmp_dir)

    skip_extraction = False
    if use_frames_count and outputdir.is_dir():
        video_frames_count = None
        if use_metadata:
            video_frames_count = metadata.get(video_frames_key, {}).get(fast_key)            
            
        if video_frames_count is None:
            video_frames_count = count_frames(video_path, fast=fast_frames_count)
            if use_metadata:
                metadata.setdefault(video_frames_key, {})[fast_key] = video_frames_count
                bl.io.save_json(metadata, metapath)
        
        bl.utils.print_verbose(
            verbose, f'Video frames count: {video_frames_count}')
        
        if use_persistent_tmp_dir:
            extracted_frames_count = count_frames_in_dir(outputdir, exclude=tmp_dir, exclude_count=tmp_dir_count)
            skip_extraction = video_frames_count == tmp_dir_count
            bl.utils.print_verbose(
                verbose, f'Frames count in temporary folder: {tmp_dir_count}')
        else:
            extracted_frames_count = count_frames_in_dir(outputdir)
        bl.utils.print_verbose(
            verbose, f'Extracted frames count: {extracted_frames_count}')
        
        if video_frames_count == extracted_frames_count:
            bl.utils.print_verbose(
                verbose, 'Frames already extracted and transformed. Skipping.')

            if use_persistent_tmp_dir:
                rm_tmp_dir()

            return
    else:
        extracted_frames_count = 0

    if not iterate:
        outputdir.mkdir(exist_ok=True, parents=True)

    if iterate:
        bl.utils.print_verbose(verbose, 'Extracting frames iteratively.')
        
        indexed_frames = enumerate(frames(video_path))
        consume(indexed_frames, extracted_frames_count) # skip frames that already exist
        
        for index, frame in indexed_frames:
            path = filename_pattern(index)
            bl.utils.print_verbose(verbose_2, f'Frame #{index} -> {bl.utils.shorten_path(path, max_len=60)} ... ', end='')
            
            if not overwrite and path.is_file():
                bl.utils.print_verbose(verbose_2, 'skip.')
                continue
            bl.utils.print_verbose(verbose_2, 'write.')

            path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(path), frame)

    elif callable_filename_pattern:
        if use_persistent_tmp_dir:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            cm = bl.utils.nullcontext(tmp_dir)
        else:
            cm = bl.utils.tempdir(prefix='_', dir=outputdir)

        with cm as temporary_folder:
            bl.utils.print_verbose(verbose, f'Extracting frames to temporary folder {bl.utils.shorten_path(temporary_folder, max_len=40)}.')
            
            intermediate_format = f'frame%d{frame_suffix}'
            intermediate_parser = f'frame{{index:d}}{frame_suffix}'

            if skip_extraction:
                bl.utils.print_verbose(verbose, f'Skipping frames extraction.')
            else:
                command_list = [
                    'ffmpeg',
                    '-i',
                    str(video_path),
                    '-qscale:v',
                    '1',
                    '-vsync',
                    'passthrough',
                    str(temporary_folder / intermediate_format)
                ]

                bl.utils.print_verbose(verbose, f'Extracting frames with command list = {command_list}')
                subprocess.run(command_list)

            source_dest_pairs = (
                (source, filename_pattern(
                    parse(intermediate_parser, source.name)['index']))
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
                    bl.utils.print_verbose(verbose_2, f'Moving {bl.utils.shorten_path(source, 30)} to {bl.utils.shorten_path(dest, 30)}.')
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    source.rename(dest)
            elif use_persistent_tmp_dir:
                rm_tmp_dir()
    else:
        if not overwrite:
            raise ValueError(
                'when not all frames are extracted, filename_pattern is not used and not in iterable mode, I have to overwrite.')

        bl.utils.print_verbose(verbose, f'Extracting frames to output folder {bl.utils.shorten_path(outputdir, max_len=40)}.')
        
        command_list = [
            'ffmpeg',
            '-i',
            f'"{video_path}"',
            '-f',
            'image2',
            f'"{bl.utils.ensure_resolved(filename_pattern, root=outputdir)}"'
        ]

        bl.utils.print_verbose(verbose, f'Extracting frames with command list = {command_list}')
        subprocess.run(command_list)

# Original command: ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
# Source: <https://stackoverflow.com/a/11175851/5811400>


def concat_videos(
        in_paths: Iterable[PathType],
        out_path: PathType
) -> None:
    # TODO: test!

    in_paths = map(Path, in_paths)
    out_path = Path(out_path)
    out_dir = out_path.mkdir(exist_ok=True, parents=True)

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
            f'"{input_file}"',
            '-c',
            'copy',
            f'"{out_path}"'
        ]
        subprocess.run(command_list)


# Original code: $ ffmpeg -i input.mp4 -c:a copy -vn -sn output.m4a
# Source: <https://superuser.com/a/633765>
def extract_audio(
        video_path: PathType,
        out_path: PathType
) -> None:
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)

    command_list = [
        'ffmpeg',
        '-i',
        f'"{video_path}"',
        '-c:a',
        'copy',
        '-vn',
        '-sn',
        f'"{out_path}"'
    ]
    subprocess.run(command_list)


@contextlib.contextmanager
def open_video(video_path: PathType) -> Iterator[cv2.VideoCapture]:
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
        recursive: bool = True,
        exclude_path: Optional[PathType] = None,
        exclude_count: Optional[Union[int, Callable[[PathType], int]]] = None
) -> int:
    path = Path(path)
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
        exclude_path = Path(exclude_path)
        if bl.utils.is_parent_dir(path, exclude_path):
            if exclude_count is None:
                exclude_count = count_frames_in_dir(exclude_path, recursive=recursive)
            elif callable(exclude_count):
                exclude_count = exclude_count(exclude_path)
            
            return n_frames_in_path - exclude_count
        else:
            return n_frames_in_path