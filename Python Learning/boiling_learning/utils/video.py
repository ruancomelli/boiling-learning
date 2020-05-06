import contextlib

# Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
# Source 2: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>
def extract_frames(filepath, outputdir, filename_pattern='frame%d.png', frame_list=None, verbose=False, make_parents=True, final_pattern=None):
    # TODO: use check_value_match?
    # TODO: raise for invalid options
    # TODO: when creating unique temporary folder, check that it can't match any generated subfolder
    # TODO: use python's tempfile module
    
    import subprocess
    
    if frame_list is not None or final_pattern is not None:
        import os
        import binascii
        from pathlib import Path
        
        from more_itertools import unzip
        from parse import parse
        
        if frame_list is not None:
            frame_list = sorted(frame_list, key=lambda x: x[0])
            
            if make_parents:
                outputdir.mkdir(exist_ok=True, parents=True)
            
            frame_indices, filenames = unzip(frame_list)
            frame_indices, filenames = list(int(x) for x in frame_indices), list(filenames)
                
            frames_used = set()
            for instance in frame_indices:
                while instance in frames_used:
                    instance += 1
                frames_used.add(instance)
                
            portion = 'select=\'' + '+'.join(f'eq(n\,{frame_index})' for frame_index in frames_used) + '\''
            
            if verbose:
                print(f'portion = {portion}')
                
        if isinstance(final_pattern, str):
            str_final_pattern = final_pattern
            final_pattern = lambda index: str_final_pattern.format(index=index)
            
        temporary_folder = outputdir / binascii.b2a_hex(os.urandom(5)).decode()
        while temporary_folder.is_dir():
            temporary_folder = outputdir / binascii.b2a_hex(os.urandom(5)).decode()
        temporary_folder.mkdir(parents=True)
        
        intermediate_format, intermediate_parser = tuple(
            f'frame{ref}{Path(filename_pattern).suffix}'
            for ref in ('%d', '{index:d}')
        )
        
        command_list =  [
            'ffmpeg',
            '-i',
            str(filepath),
            '-qscale:v',
            '1'
        ] + (
            [
                '-vf',
                portion,
            ]
            if frame_list is not None
            else []
        ) + [
            '-vsync',
            '0',
            str(temporary_folder / intermediate_format)
        ]
        
        if verbose:
            print(f'command list = {command_list}')
        
        subprocess.run(command_list)

        if final_pattern is not None:
            source_dest_pairs = (
                (source, outputdir / final_pattern(parse(intermediate_parser, source.name)['index']))
                for source in temporary_folder.iterdir()
            )
        else:
            source_dest_pairs = (
                (temporary_folder / (intermediate_format % i), outputdir / filename)
                for i, filename in enumerate(filenames, start=1)
            )
        source_dest_pairs = filter(lambda source_dest_pair: source_dest_pair[0].is_file(), source_dest_pairs)
            
        for source, dest in source_dest_pairs:            
            if make_parents:
                dest.parent.mkdir(exist_ok=True, parents=True)
            source.rename(dest)
                
        temporary_folder.rmdir()
    else:
        if make_parents:
            outputdir.mkdir(exist_ok=True, parents=True)
        
        command_list =  [
            'ffmpeg',
            '-i',
            f'"{filepath}"',
            '-f',
            'image2',
            f'"{outputdir / filename_pattern}"'
        ]
        subprocess.run(command_list)
        
    
# Original code: $ ffmpeg -i input.mp4 -c:a copy -vn -sn output.m4a
# Source: <https://superuser.com/a/633765>
def extract_audio(in_path, out_path):
    import subprocess

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
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    
    try:
        yield cap
    finally:
        cap.release()
        
# Does not work with GOPRO format
def frames(video_path):
    with open_video(video_path) as cap:
        while cap.grab():
            flag, frame = cap.retrieve()
            yield frame
            
def count_frames(video_path, fast=False):
    import cv2

    if fast:
        with open_video(video_path) as cap:
            return int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    else:
        from more_itertools import ilen
        
        return ilen(frames(video_path))
