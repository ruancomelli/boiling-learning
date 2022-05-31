import typing
from typing import Iterable, Iterator, Optional, Tuple

import h5py
import hdf5plugin
import more_itertools as mit
import numpy as np
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import Video, VideoFrame, open_video
from boiling_learning.utils.utils import PathLike, resolve


class HDF5VideoSliceableDataset(SliceableDataset[VideoFrame]):
    def __init__(self, filepath: PathLike, dataset_name: str) -> None:
        self._file = h5py.File(str(resolve(filepath)), 'r', swmr=True)
        self._dataset_name = dataset_name

    def getitem_from_index(self, index: int) -> VideoFrame:
        return typing.cast(VideoFrame, self.dataset()[index])

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[VideoFrame, ...]:
        return tuple(self.dataset() if indices is None else self.dataset()[list(indices)])

    def __enter__(self) -> None:
        self._file.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.__exit__(exc_type, exc_value, traceback)

    def dataset(self) -> h5py.Dataset:
        return self._file[self._dataset_name]


def video_to_hdf5(
    video: Video,
    dest: PathLike,
    *,
    dataset_name: str,
    batch_size: int = 1,
    transformers: Tuple[Transformer[VideoFrame, VideoFrame], ...] = (),
) -> None:
    """Save video as an HDF5 file

    Arguments:
        transformers: a tuple of transformers mapping a batch of frames to
        another batch of frames. Suitable for broadcasting image transformers.
    """
    destination = str(resolve(dest))

    # use OpenCV reader because it is much more memory efficient than PIMS
    frames = video_frames(video.path)
    frame_chunks = mit.chunked(frames, batch_size)
    array_chunks = map(np.array, frame_chunks)

    with h5py.File(destination, 'a') as file:
        dataset = file.require_dataset(
            dataset_name,
            video.shape,
            dtype='f',
            # best compression algorithm I found - good compression, fastest decompression
            **hdf5plugin.LZ4(),
        )
        for index, batch in enumerate(array_chunks):
            start, end = index * batch_size, (index + 1) * batch_size

            logger.debug(f'Writing frames {start}:{end} from {video.path} to {destination}')

            for transformer in transformers:
                batch = map(transformer, batch)

            batch = np.array(batch)

            dataset.write_direct(batch, dest_sel=np.s_[start:end])
            dataset.flush()
            file.flush()


def video_frames(video_path: PathLike) -> Iterator[VideoFrame]:
    with open_video(video_path) as cap:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield frame
