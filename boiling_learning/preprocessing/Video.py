from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pims

from boiling_learning.utils.pathutils import PathLike
from boiling_learning.utils.utils import ensure_resolved


class Video(Sequence[np.ndarray]):
    def __init__(
        self,
        path: PathLike,
    ) -> None:
        self.path: Path = ensure_resolved(path)

        self.video: Optional[pims.Video] = None
        self._is_open_video: bool = False
        self.start: int = 0
        self.end: Optional[int] = None

    def __len__(self) -> int:
        self.open_video()
        return self.end - self.start

    def __getitem__(self, index: int) -> np.ndarray:
        self.open_video()

        length = len(self)

        if index >= length:
            raise IndexError(f'index must be less than {length}')

        absolute_index = self.start + index

        # normalize frames to TF's standard: `pims.Video` returns frames
        # with float datatype, but scaled from 0 to 255 whereas TF expects
        # those floating point types to be between 0 and 1
        return self.at(absolute_index)

    def at(self, index: int) -> np.ndarray:
        return self.video[index] / 255

    def open_video(self) -> None:
        if not self._is_open_video:
            self.video = pims.Video(str(self.path))

            if self.end is None:
                self.end = len(self.video)

            self._is_open_video = True

    def close_video(self) -> None:
        if self._is_open_video:
            self.video.close()
        self.video = None
        self._is_open_video = False
