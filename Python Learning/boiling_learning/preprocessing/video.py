import contextlib
import subprocess
import os
import binascii
import operator
import tempfile
from pathlib import Path
        
import cv2
from more_itertools import ilen, unzip
from parse import parse

import boiling_learning as bl

# Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
# Source 2: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>
def extract_frames(
    filepath,
    outputdir,
    filename_pattern='frame%d.png',
    verbose=False,
    final_pattern=None,
    fast_frames_count=None,
    iterate=False,
    overwrite=False
):
    # TODO: use check_value_match?
    # TODO: raise for invalid options
    
    if fast_frames_count is not None and overwrite:
        raise ValueError('cannot overwrite when frames count mode is passed.')
    
    bl.utils.print_verbose(verbose, f'Extracting frames from {filepath} to {outputdir}')
    
    filepath = Path(filepath)
    outputdir = Path(outputdir)
    frame_suffix = Path(filename_pattern).suffix 

    if fast_frames_count is not None and outputdir.is_dir():
        video_frames_count = count_frames(filepath, fast=fast_frames_count)
        extracted_frames_count = ilen(outputdir.rglob(f'*{frame_suffix}'))
        
        bl.utils.print_verbose(verbose, f'Frames extracted: {extracted_frames_count}/{video_frames_count}')
        
        if video_frames_count == extracted_frames_count:
            bl.utils.print_verbose(verbose, 'Frames already extracted. Skipping.')
            return
    
    outputdir.mkdir(exist_ok=True, parents=True)
    
    if final_pattern is not None:
        if isinstance(final_pattern, str):
            formatter = final_pattern.format
            
            def final_pattern(index):
                return formatter(index=index)                
            
    if iterate: # TODO: test!!!
        for index, frame in enumerate(frames(filepath)):
            path = outputdir / final_pattern(index)

            if not overwrite and path.is_file():
                continue

            path.parent.mkdir(exist_ok=True, parents=True)            
            cv2.imwrite(str(path), frame)
            
    elif final_pattern is not None:
        with bl.utils.tempdir(prefix='_', dir=outputdir) as temporary_folder:
            intermediate_format = f'frame%d{frame_suffix}'
            intermediate_parser = f'frame{{index:d}}{frame_suffix}'
            
            command_list =  [
                'ffmpeg',
                '-i',
                str(filepath),
                '-qscale:v',
                '1',
                '-vsync',
                '0',
                str(temporary_folder / intermediate_format)
            ]
            
            bl.utils.print_verbose(verbose, f'command list = {command_list}')
            
            subprocess.run(command_list)

            source_dest_pairs = (
                (source, outputdir / final_pattern(parse(intermediate_parser, source.name)['index']))
                for source in temporary_folder.iterdir()
            )
            
            source_dest_pairs = filter(lambda source_dest_pair: source_dest_pair[0].is_file(), source_dest_pairs)
            
            if not overwrite:
                source_dest_pairs = filter(lambda source_dest_pair: not source_dest_pair[1].is_file(), source_dest_pairs)
                
            for source, dest in source_dest_pairs:            
                dest.parent.mkdir(exist_ok=True, parents=True)
                source.rename(dest)
    else:
        if not overwrite:
            raise ValueError('when not all frames are extracted, final_pattern is not used and not in iterable mode, I have to overwrite.')
        
        command_list =  [
            'ffmpeg',
            '-i',
            f'"{filepath}"',
            '-f',
            'image2',
            f'"{outputdir / filename_pattern}"'
        ]
        subprocess.run(command_list)
        
# Original command: ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
# Source: <https://stackoverflow.com/a/11175851/5811400>
def concat_videos(in_paths, out_path):
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
            
        command_list =  [
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
def extract_audio(in_path, out_path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
        
    command_list =  [
        'ffmpeg',
        '-i',
        f'"{in_path}"',
        '-c:a',
        'copy',
        '-vn',
        '-sn',
        f'"{out_path}"'
    ]
    subprocess.run(command_list)

@contextlib.contextmanager
def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    
    try:
        yield cap
    finally:
        cap.release()
        
# Does not work with GOPRO format
def frames(video_path, suppress_retrieval_failure=True):
    with open_video(video_path) as cap:
        while cap.grab():
            flag, frame = cap.retrieve()
            if flag:
                yield frame
            elif not suppress_retrieval_failure:
                raise RuntimeError(f'failed frame retrieval for video at {video_path}')
            
def count_frames(video_path, fast=False):
    if fast:
        with open_video(video_path) as cap:
            return int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    else:        
        return ilen(frames(video_path))
