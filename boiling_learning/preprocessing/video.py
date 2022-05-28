import contextlib
import subprocess
import typing
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

import cv2
import h5py
import hdf5plugin
import more_itertools as mit
import numpy as np
import pims
from imageio.core import CannotReadFrameError
from loguru import logger

from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.utils import PathLike, resolve

# VideoFrame = npt.NDArray[np.float32]
VideoFrame = np.ndarray


def convert_video(
    in_path: PathLike,
    out_path: PathLike,
    remove_audio: bool = False,
    fps: Optional[Union[str, int, float]] = None,
    overwrite: bool = False,
) -> None:
    # For `fps`, see <https://superuser.com/a/729351>.

    in_path = resolve(in_path)
    out_path = resolve(out_path, parents=True)

    logger.info(f'Converting video: {in_path} -> {out_path}')

    if overwrite and out_path.is_file():
        logger.info(f'Overwriting {out_path}')
        out_path.unlink()
        # TO-DO: in Python 3.8, use out_path.unlink(missing_ok=True) and remove one condition

    if out_path.is_file():
        logger.info(f'Destination file already exists. Skipping video conversion: {out_path}')
    else:
        command_list = ['ffmpeg', '-i', str(in_path), '-vsync', '0']
        if remove_audio:
            command_list.append('-an')
        if fps is not None:
            command_list.extend(['-r', str(fps)])
        command_list.append(str(out_path))

        logger.debug('Command list =', command_list)

        logger.info('Running conversion...')
        subprocess.run(command_list)
        logger.info('Succesfully converted video')


@contextlib.contextmanager
def open_video(video_path: PathLike) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(str(resolve(video_path)))

    try:
        yield cap
    finally:
        cap.release()


def get_fps(video_path: PathLike) -> float:
    with open_video(video_path) as cap:
        return typing.cast(float, cap.get(cv2.CAP_PROP_FPS))


def video_frames(video_path: PathLike) -> Iterator[VideoFrame]:
    with open_video(video_path) as cap:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield frame


class OpenVideoError(Exception):
    pass


class Video(Sequence[VideoFrame]):
    def __init__(self, path: PathLike) -> None:
        self.path = resolve(path)
        self._video: Optional[pims.Video] = None
        self._should_shrink_to_valid_end_frames: bool = True

    @property
    def video(self) -> pims.Video:
        return self.open()

    @video.setter
    def video(self, video: pims.Video) -> pims.Video:
        self._video = video

    def __getitem__(self, key: int) -> VideoFrame:
        return self.open()[key] / 255

    def __len__(self) -> int:
        return len(self.open())

    def open(self) -> pims.Video:
        if not self.is_open():
            try:
                self._video = pims.Video(str(self.path))
            except Exception as e:
                raise OpenVideoError(f'Error while opening video {self.path}') from e

        return self._video

    def close(self) -> None:
        if self.is_open():
            self._video.close()
            self._video = None

    def is_open(self) -> bool:
        return self._video is not None

    @property
    def shape(self) -> Tuple[int, ...]:
        try:
            return (len(self), *self[0].shape)
        except IndexError:
            return ()


def video_to_hdf5(video: Video, dest: PathLike, *, dataset_name: str, batch_size: int = 1) -> None:
    destination = str(resolve(dest))

    # use OpenCV reader because it is much more memory efficient than PIMS
    frames = video_frames(video.path)
    frame_chunks = mit.chunked(frames, batch_size)
    array_chunks = map(np.array, frame_chunks)

    with h5py.File(destination, 'a') as file:
        for index, batch in enumerate(array_chunks):
            dataset = file.require_dataset(
                dataset_name,
                video.shape,
                dtype='f',
                # best compression algorithm I found - good compression, fastest decompression
                **hdf5plugin.LZ4(),
            )
            start, end = index * batch_size, (index + 1) * batch_size
            dataset.write_direct(batch, dest_sel=np.s_[start:end])
            dataset.flush()
            file.flush()


def get_frame_from_hdf5(path: PathLike, index: int, *, dataset_name: str) -> VideoFrame:
    resolved = str(resolve(path))

    with h5py.File(resolved) as f:
        return typing.cast(VideoFrame, f[dataset_name][index])


@json.encode.instance(Video)
def _encode_video(obj: Video) -> json.JSONDataType:
    return json.serialize(obj.path)


@describe.instance(Video)
def _describe_video(obj: Video) -> Path:
    return obj.path


@serialize.instance(VideoFrame)
def _serialize_video_frame(instance: VideoFrame, path: Path) -> None:
    np.save(path.with_suffix('.npy'), instance)


@deserialize.dispatch(VideoFrame)
def _deserialize_video_frame(path: Path, metadata: Metadata) -> VideoFrame:
    return np.load(path.with_suffix('.npy'))


def shrink_to_valid_end_frames(video: Video) -> None:
    logger.debug(f"Shrinking video to valid end frame at \"{video.path}\"")

    video.video = video.video[: valid_end_frame(video) + 1]

    logger.debug(f"Video kept with {len(video)} frames at \"{video.path}\"")


def valid_end_frame(video: Video) -> int:
    logger.debug(f"Searching for valid end frame for video at \"{video.path}\"")

    for index in reversed(range(len(video))):
        # the following exceptions are "expected" and signify that this candidate end frame is
        # invalid
        with contextlib.suppress(CannotReadFrameError, RuntimeError, AttributeError):
            # try to access frame at `index`
            video[index]
            # if access is successful (by not raising any errors), we found a valid end frame
            logger.debug(f"Valid end frame is {index} for video at \"{video.path}\"")
            return index
    return -1
