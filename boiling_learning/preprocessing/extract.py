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


class NoFramesError(Exception):
    pass


class ExtractedFramesDataset(SequenceSliceableDataset[Image]):
    def __init__(self, path: PathLike) -> None:
        self.path = resolve(path)

        if not _is_done_extracting(self.path):
            raise NoFramesError

        super().__init__(
            imread_collection(
                str(self.path / '*.png'),
                conserve_memory=True,  # if I ever need caching, use a memory cache
            )
        )

    @classmethod
    def from_video(cls, video: Video, directory: PathLike, /) -> ExtractedFramesDataset:
        # Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
        # Source 2: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>
        directory = resolve(directory, dir=True)

        if not _is_done_extracting(directory):
            logger.info(f'Extracting frames from {video.path} to {directory}')

            number_of_digits = len(str(len(video)))
            filename_pattern = f'%{number_of_digits}d.png'

            try:
                (
                    ffmpeg.input(str(video.path))
                    .output(str(directory / filename_pattern), format='image2')
                    .run()
                )
            except Exception as e:
                raise ExtractionError from e

            _mark_as_done_extracting(directory)

        return ExtractedFramesDataset(directory)


def _is_done_extracting(directory: Path) -> bool:
    return _done_path_for_directory(directory).exists()


def _mark_as_done_extracting(directory: Path) -> bool:
    return _done_path_for_directory(directory).touch(exist_ok=True)


def _done_path_for_directory(directory: Path) -> Path:
    return directory / '__done__'
