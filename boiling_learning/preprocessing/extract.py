from __future__ import annotations

import typing
from collections.abc import Iterable, Iterator
from pathlib import Path

import ffmpeg
import imageio
import numpy as np
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.image_datasets import Image, Images
from boiling_learning.preprocessing.video import Video
from boiling_learning.utils.pathutils import PathLike, resolve

DEFAULT_EMPTY_FRAME_THRESHOLD = 1e-6


class ExtractionError(RuntimeError):
    pass


class LoadImageError(RuntimeError):
    pass


class ExtractedFramesDataset(SliceableDataset[Image]):
    """A dataset composed of extracted frames.

    Args:
        eager: in non-eager mode, every call to `fetch` will extract the required
            frames. In eager mode, calls to `fetch` will extract _all_ possible frames.
            Use eager mode when you know that all frames will be used since it is much
            more efficient to extract all frames at once.
        robust: occasionally the frame extraction procedure may fail, yielding
            completely black images. In the `robust` mode, such frames are automatically
            deleted and re-extracted during the fetching step.
    """

    def __init__(
        self,
        video_path: PathLike,
        path: PathLike,
        /,
        *,
        length: int | None = None,
        eager: bool = False,
        robust: bool = True,
    ) -> None:
        self.video_path = resolve(video_path)
        self.path = resolve(path, dir=True)

        self._eager = eager
        self._robust = robust

        self._length = length if length is not None else len(Video(self.video_path))

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return f"ExtractedFramesDataset({self.video_path}, {self.path})"

    def getitem_from_index(self, index: int) -> Image:
        return self.fetch((index,))[0]

    def fetch(self, indices: Iterable[int] | None = None) -> Images:
        all_indices = range(len(self))
        indices = tuple(all_indices if indices is None else indices)
        unique_indices = frozenset(indices)

        if missing_indices := frozenset(
            filter(self._is_missing, all_indices if self._eager else unique_indices)
        ):
            self._extract_frames(missing_indices)

        for index in unique_indices:
            if self._is_missing(index):
                raise ExtractionError(f"failed to extract frame #{index} for {self}")

        return np.stack(
            list(self._robust_fetch_frames(indices))
            if self._robust
            else [self._load_frame(index) for index in indices]
        )

    def _robust_fetch_frames(self, indices: tuple[int, ...], /) -> Iterator[Image]:
        unique_indices = frozenset(indices)
        indexed_frames = {
            index: self._maybe_load_frame(index) for index in unique_indices
        }

        if failed_indices := tuple(
            index for index, frame in indexed_frames.items() if frame is None
        ):
            for failed_index in failed_indices:
                self._index_to_path(failed_index).unlink()
            self._extract_frames(failed_indices)

            renewed_frames = {
                index: self._maybe_load_frame(index) for index in failed_indices
            }

            if failed_indices := tuple(
                index for index, frame in renewed_frames.items() if frame is None
            ):
                raise RuntimeError(
                    f"Failed generate frames {failed_indices} in robust mode for {self}"
                )

            indexed_frames |= renewed_frames

        return (
            # we checked before that no image is `None`
            typing.cast(Image, indexed_frames[index])
            for index in indices
        )

    def _maybe_load_frame(self, index: int, /) -> Image | None:
        try:
            return self._load_frame(index)
        except LoadImageError:
            return None

    def _load_frame(self, index: int, /) -> Image:
        path = self._index_to_path(index)
        try:
            frame = imageio.imread(path)
        except (ValueError, RuntimeError) as e:
            raise LoadImageError(f"failed to load frame #{index} for {self}") from e

        if self._robust and is_empty_frame(frame):
            raise LoadImageError(f"frame #{index} for {self} is empty")

        return frame

    def _is_missing(self, index: int) -> bool:
        return not self._index_to_path(index).exists()

    def _index_to_path(self, index: int) -> Path:
        return self.path / self._index_to_filename(index)

    def _index_to_filename(self, index: int) -> str:
        number_of_digits = self._filename_number_of_digits()
        index_str = str(index)
        leading_zeros = "0" * (number_of_digits - len(index_str))
        return f"{leading_zeros}{index_str}{self._filename_suffix()}"

    def _extract_frames(self, indices: Iterable[int]) -> None:
        # Original code:
        # 1: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
        # 2: $ ffmpeg -i in.mp4 -vf select='eq(n\,100)+eq(n\,184)+eq(n\,213)' -vsync 0 frames%d.jpg
        # Source 1: https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818
        # Source 2:
        # https://stackoverflow.com/questions/38253406/extract-list-of-specific-frames-using-ffmpeg
        indices = sorted(frozenset(indices))

        if not indices:
            logger.info(
                f"Skipping frame extraction for {self} since no indices were provided"
            )
            return

        logger.info(f"Extracting frames {indices} for {self}")

        filename_pattern = (
            f"%{self._filename_number_of_digits()}d{self._filename_suffix()}"
        )

        try:
            (
                ffmpeg.input(str(self.video_path))
                .filter("select", "+".join(f"eq(n,{index})" for index in indices))
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
        return ".png"


def is_empty_frame(
    frame: Image,
    /,
    *,
    threshold: float = DEFAULT_EMPTY_FRAME_THRESHOLD,
) -> bool:
    return float(np.absolute(frame).mean()) < threshold
