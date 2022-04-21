import contextlib
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import cv2
import numpy as np
import pims
from imageio.core import CannotReadFrameError
from loguru import logger

from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.utils import PathLike, resolve
from boiling_learning.utils.descriptions import describe

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
    video_path = resolve(video_path)
    cap = cv2.VideoCapture(str(video_path))

    try:
        yield cap
    finally:
        cap.release()


def get_fps(video_path: PathLike) -> float:
    with open_video(video_path) as cap:
        return cap.get(cv2.CAP_PROP_FPS)


class OpenVideoError(Exception):
    pass


class Video(Sequence[VideoFrame]):
    def __init__(self, path: PathLike) -> None:
        self.path: Path = resolve(path)
        self._video: Optional[pims.Video] = None

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
                raise OpenVideoError(f'Error while opening video {self.path}:\n{e}') from e

            self._shrink_to_valid_end_frames()

        return self._video

    def close(self) -> None:
        if self.is_open():
            self._video.close()
            self._video = None

    def is_open(self) -> bool:
        return self._video is not None

    def _shrink_to_valid_end_frames(self) -> None:
        valid_end_frame = self._valid_end_frame()
        self.video = self.video[: valid_end_frame + 1]

    def _valid_end_frame(self) -> int:
        for index in reversed(range(len(self))):
            # the following exceptions are "expected" and signify that this candidate end frame is
            # invalid
            with contextlib.suppress(CannotReadFrameError, RuntimeError, AttributeError):
                # try to access frame at `index`
                self[index]
                # if access is successful (by not raising any errors), we found a valid end frame
                return index
        return -1


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
