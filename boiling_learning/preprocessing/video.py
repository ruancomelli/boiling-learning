from __future__ import annotations

import functools
import gc
import os
import subprocess
import typing
from pathlib import Path
from types import TracebackType
from typing import Iterable, Iterator, Optional, Type, Union

import decord
import numpy as np
import numpy.typing as npt
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.utils.pathutils import PathLike, resolve

if typing.TYPE_CHECKING:
    VideoFrameU8 = npt.NDArray[np.uint8]
    VideoFrameF32 = npt.NDArray[np.float32]
    VideoFrame = Union[VideoFrameU8, VideoFrameF32]
else:
    VideoFrameU8 = np.ndarray
    VideoFrameF32 = np.ndarray
    VideoFrame = np.ndarray

# there is no shape in nympy yet, so we can't really differentiate one frame from many
VideoFrames = VideoFrame


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


class Video(SliceableDataset[VideoFrame]):
    def __init__(self, path: PathLike) -> None:
        # workaround for limiting memory usage
        os.environ['DECORD_EOF_RETRY_MAX'] = '128'

        self._path = resolve(path)
        self._video: decord.VideoReader | None = None

    @property
    def path(self) -> Path:
        return self._path

    def getitem_from_index(self, index: int) -> VideoFrameU8:
        with self as frames:
            return typing.cast(VideoFrameU8, frames[index].asnumpy())

    def fetch(self, indices: Optional[Iterable[int]] = None) -> tuple[VideoFrameU8, ...]:
        indices = range(len(self)) if indices is None else list(indices)

        with self as frames:
            return tuple(frames.get_batch(indices).asnumpy())

    def __iter__(self) -> Iterator[VideoFrameU8]:
        with self as frames:
            for frame in frames:
                yield frame.asnumpy()

    @functools.cache
    def __len__(self) -> int:
        with self as frames:
            return len(frames)

    @functools.cache
    def fps(self) -> float:
        with self as frames:
            return typing.cast(float, frames.get_avg_fps())

    def __enter__(self) -> decord.VideoReader:
        if self._video is None:
            try:
                self._video = decord.VideoReader(str(self.path))
                # workaround for limiting memory usage
                self._video.seek(0)
            except Exception as e:
                raise OpenVideoError(f'Error while opening video {self.path}') from e

        return self._video

    def __exit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        if self._video is not None:
            # workaround for limiting memory usage
            self._video.seek(0)
            del self._video
            gc.collect()
            self._video = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path})'


class OpenVideoError(Exception):
    pass


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
