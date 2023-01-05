from __future__ import annotations

from pathlib import Path

import ffmpeg
from loguru import logger
from skimage.io import imread_collection

from boiling_learning.datasets.sliceable import SequenceSliceableDataset
from boiling_learning.image_datasets import Image
from boiling_learning.preprocessing.video import Video
from boiling_learning.utils.pathutils import PathLike, resolve


class ExtractionError(Exception):
    pass


class ExtractedFramesDataset(SequenceSliceableDataset[Image]):
    def __init__(self, video: Video, path: PathLike, /) -> None:
        self.path = resolve(path, dir=True)
        self.video = video

        if not self._is_done_extracting():
            self._extract_frames()

        super().__init__(
            imread_collection(
                str(self.path / '*.png'),
                conserve_memory=True,  # if I ever need caching, use a memory cache
            )
        )

    def _extract_frames(self) -> None:
        # Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
        # Source: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>
        logger.info(f'Extracting frames from {self.video.path} to {self.path}')

        number_of_digits = len(str(len(self.video)))
        filename_pattern = f'%{number_of_digits}d.png'

        try:
            (
                ffmpeg.input(str(self.video.path))
                .output(str(self.path / filename_pattern), format='image2')
                .run()
            )
        except Exception as e:
            raise ExtractionError from e

        self._mark_as_done_extracting()

    def _is_done_extracting(self) -> bool:
        return _done_path_for_directory(self.path).exists()

    def _mark_as_done_extracting(self) -> None:
        _done_path_for_directory(self.path).touch(exist_ok=True)


def _done_path_for_directory(directory: Path, /) -> Path:
    return directory / '__done__'
