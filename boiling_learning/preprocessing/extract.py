from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import ffmpeg
import imageio
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.image_datasets import Image
from boiling_learning.preprocessing.video import Video
from boiling_learning.utils.pathutils import PathLike, resolve


class ExtractionError(RuntimeError):
    pass


class LoadImageError(RuntimeError):
    pass


class ExtractedFramesDataset(SliceableDataset[Image]):
    '''A dataset composed of extracted frames.

    Args:
        eager: in non-eager mode, every call to `fetch` will extract the required
            frames. In eager mode, calls to `fetch` will extract _all_ possible frames.
            Use eager mode when you know that all frames will be used since it is much
            more efficient to extract all frames at once.
    '''

    def __init__(
        self,
        video: Video,
        path: PathLike,
        /,
        *,
        eager: bool = False,
    ) -> None:
        self.path = resolve(path, dir=True)
        self.video = video
        self._eager = eager

    def __len__(self) -> int:
        return len(self.video)

    def __repr__(self) -> str:
        return f'ExtractedFramesDataset({self.video.path}, {self.path})'

    def getitem_from_index(self, index: int) -> Image:
        return self.fetch((index,))[0]

    def fetch(self, indices: Iterable[int] | None = None) -> tuple[Image, ...]:
        all_indices = range(len(self.video))
        indices = tuple(all_indices if indices is None else indices)

        if missing_indices := sorted(
            filter(self._is_missing, all_indices if self._eager else indices)
        ):
            self._extract_frames(missing_indices)

        for index in indices:
            if self._is_missing(index):
                raise ExtractionError(f'failed to extract frame #{index}')

        return tuple(self._load_frame(index) for index in indices)

    def _load_frame(self, index: int) -> Image:
        path = self._index_to_path(index)
        try:
            return imageio.imread(path)
        except (ValueError, RuntimeError) as e:
            raise LoadImageError(f'failed to load frame #{index} from {path}') from e

    def _is_missing(self, index: int) -> bool:
        return not self._index_to_path(index).exists()

    def _index_to_path(self, index: int) -> Path:
        return self.path / self._index_to_filename(index)

    def _index_to_filename(self, index: int) -> str:
        number_of_digits = self._filename_number_of_digits()
        index_str = str(index)
        leading_zeros = '0' * (number_of_digits - len(index_str))
        return f'{leading_zeros}{index_str}{self._filename_suffix()}'

    def _extract_frames(self, indices: Iterable[int]) -> None:
        # Original code:
        # 1: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
        # 2: $ ffmpeg -i in.mp4 -vf select='eq(n\,100)+eq(n\,184)+eq(n\,213)' -vsync 0 frames%d.jpg
        # Source 1: https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818
        # Source 2:
        # https://stackoverflow.com/questions/38253406/extract-list-of-specific-frames-using-ffmpeg
        indices = tuple(indices)
        logger.info(f'Extracting frames {indices} from {self.video.path} to {self.path}')

        filename_pattern = f'%{self._filename_number_of_digits()}d{self._filename_suffix()}'

        try:
            (
                ffmpeg.input(str(self.video.path))
                .filter('select', '+'.join(f'eq(n,{index})' for index in indices))
                .output(
                    str(self.path / filename_pattern),
                    frame_pts=True,  # saves each frame with their index as their name
                    vsync=0,  # avoids duplicates
                )
                .run()
            )
        except Exception as e:
            raise ExtractionError from e

    def _filename_number_of_digits(self) -> int:
        return len(str(len(self)))

    def _filename_suffix(self) -> str:
        return '.png'
