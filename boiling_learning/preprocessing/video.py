import contextlib
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import cv2
import numpy as np
import numpy.typing as npt
import pims
from imageio.core import CannotReadFrameError

from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.utils import PathLike, VerboseType, resolve, shorten_path

VideoFrame = npt.NDArray[np.float32]


def convert_video(
    in_path: PathLike,
    out_path: PathLike,
    remove_audio: bool = False,
    fps: Optional[Union[str, int, float]] = None,
    verbose: VerboseType = False,
    overwrite: bool = False,
) -> None:
    # For `fps`, see <https://superuser.com/a/729351>.

    in_path = resolve(in_path)
    out_path = resolve(out_path, parents=True)

    if verbose:
        print(
            'Converting video',
            shorten_path(in_path, max_len=50),
            '->',
            shorten_path(out_path, max_len=50),
        )

    if overwrite and out_path.is_file():
        if verbose:
            print('Overwriting', shorten_path(out_path, max_len=50))
        out_path.unlink()
        # TO-DO: in Python 3.8, use out_path.unlink(missing_ok=True) and remove one condition

    if out_path.is_file():
        if verbose:
            print('Destination file already exists. Skipping video conversion.')
    else:
        command_list = ['ffmpeg', '-i', str(in_path), '-vsync', '0']
        if remove_audio:
            command_list.append('-an')
        if fps is not None:
            command_list.extend(['-r', str(fps)])
        command_list.append(str(out_path))

        if verbose:
            print(
                'Converting video:',
                shorten_path(in_path, max_len=40),
                '->',
                shorten_path(out_path, max_len=40),
            )
            print('Command list =', command_list)

        subprocess.run(command_list)


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


class Video(Sequence[VideoFrame]):
    def __init__(self, path: PathLike) -> None:
        self.path: Path = resolve(path)
        self.video: Optional[pims.Video] = None

    def __getitem__(self, key: int) -> VideoFrame:
        return self.open()[key] / 255

    def __len__(self) -> int:
        return len(self.open())

    def open(self) -> pims.Video:
        if not self.is_open():
            self.video = pims.Video(str(self.path))
            self._shrink_to_valid_end_frames()

        return self.video

    def close(self) -> None:
        if self.is_open():
            self.video.close()
            self.video = None

    def is_open(self) -> bool:
        return self.video is not None

    def _shrink_to_valid_end_frames(self) -> None:
        valid_end_frame = self._valid_end_frame()
        self.video = self.video[: valid_end_frame + 1]

    def _valid_end_frame(self) -> int:
        for index in range(
            len(self) - 1,  # starting from the last frame
            -1,  # up until the first frame (frame 0 is included)
            -1,  # decrementing this index
        ):
            # the following exceptions are "expected" and signify that this candidate end frame is
            # invalid
            with contextlib.suppress(CannotReadFrameError, RuntimeError, AttributeError):
                # try to access frame at `index`
                self[index]
                # if access is successful (by not raising any errors), we found a valid end frame
                return index
        return -1


@describe.instance(Video)
def _describe_video(obj: Video) -> Path:
    return obj.path
