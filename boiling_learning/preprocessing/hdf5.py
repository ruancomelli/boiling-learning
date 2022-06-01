import typing
from typing import Iterable, Iterator, Optional, Tuple

import funcy
import h5py
import hdf5plugin
import more_itertools as mit
import numpy as np
from loguru import logger
from typing_extensions import Literal

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import Video, VideoFrame, open_video
from boiling_learning.utils.utils import PathLike, resolve, unsort


class HDF5VideoSliceableDataset(SliceableDataset[VideoFrame]):
    def __init__(self, filepath: PathLike, dataset_name: str) -> None:
        self._filepath = resolve(filepath, parents=True)
        self._file = h5py.File(str(self._filepath), 'r', swmr=True)
        self._dataset_name = dataset_name
        self._is_open: bool = False

    def __len__(self) -> int:
        return len(self.dataset())

    def getitem_from_index(self, index: int) -> VideoFrame:
        return typing.cast(VideoFrame, self.dataset()[index])

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[VideoFrame, ...]:
        logger.debug(f'Fetching frames {indices} from {self}')

        if indices is None:
            return tuple(self.dataset())

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        frames = self.dataset()[list(sorted_indices)]
        return tuple(frames[unsorter] for unsorter in unsorters)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dataset_name}@{self._filepath})'

    def __enter__(self) -> None:
        self._file.__enter__()
        self._is_open = True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._is_open = False
        self._file.__exit__(exc_type, exc_value, traceback)

    def dataset(self) -> h5py.Dataset:
        if not self._is_open:
            self.__enter__()

        return self._file[self._dataset_name]

    def close(self) -> None:
        self._file.close()


def video_to_hdf5(
    video: Video,
    dest: PathLike,
    *,
    dataset_name: str,
    batch_size: int = 1,
    transformers: Tuple[Transformer[VideoFrame, VideoFrame], ...] = (),
    open_mode: Literal['w', 'a'] = 'a',
) -> None:
    """Save video as an HDF5 file."""
    destination = str(resolve(dest))

    composed_transformer = funcy.rcompose(*transformers)
    example_frame = composed_transformer(video[0])

    # use OpenCV reader because it is much more memory efficient than PIMS
    frames = video_frames(video.path)
    frame_chunks = mit.chunked(frames, batch_size)
    array_chunks = map(np.array, frame_chunks)

    with h5py.File(destination, open_mode) as file:
        dataset = file.require_dataset(
            dataset_name,
            (len(video), *example_frame.shape),
            dtype='f',
            # best compression algorithm I found - good compression, fastest decompression
            **hdf5plugin.LZ4(),
        )
        for index, batch in enumerate(array_chunks):
            start = index * batch_size
            end = start + len(batch)

            batch = np.array(
                [composed_transformer(frame) / 255 for frame in batch], dtype=batch.dtype
            )
            logger.debug(f'Writing frames {start}:{end} from {video.path} to {destination}')

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
