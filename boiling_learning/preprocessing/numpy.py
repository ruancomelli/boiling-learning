from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Sequence, Tuple, TypeVar

import numpy as np
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.utils import PathLike, resolve, unsort

_Array = TypeVar('_Array', bound=np.ndarray)


class NumpyMemoryMappedSliceableDataset(SliceableDataset[_Array]):
    def __init__(self, filepath: PathLike) -> None:
        self._filepath = resolve(filepath, parents=True)

    @contextmanager
    def open(self) -> Iterator[Sequence[_Array]]:
        yield np.memmap(self._filepath, mode='r')  # type: ignore

    def __len__(self) -> int:
        with self.open() as data:
            return len(data)

    def getitem_from_index(self, index: int) -> _Array:
        with self.open() as data:
            return data[index]

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[_Array, ...]:
        if indices is None:
            with self.open() as data:
                return tuple(data)

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        sorted_indices = list(sorted_indices)
        with self.open() as data:
            frames = data[sorted_indices]
            return tuple(frames[unsorter] for unsorter in unsorters)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._filepath})'


def frames_to_numpy(
    frames: SliceableDataset[VideoFrame],
    dest: PathLike,
    *,
    buffer_size: Optional[int] = None,
) -> None:
    """Save frames as an HDF5 file."""
    destination = resolve(dest, parents=True)

    number_of_frames = len(frames)
    first_frame = frames[0]

    fp = np.memmap(
        destination,
        mode='w+',
        dtype=first_frame.dtype,
        shape=(number_of_frames, *first_frame.shape),
    )

    if buffer_size is None:
        logger.debug(f'Writing ALL frames ({number_of_frames}) to {destination}')

        fp[:] = frames.fetch()

    else:
        for chunk_index, frames_batch in enumerate(frames.batch(buffer_size)):
            n_frames = len(frames_batch)
            start = chunk_index * buffer_size
            end = start + n_frames

            logger.debug(f'Writing frames {start}:{end} ({n_frames}) to {destination}')

            fp[start:end] = frames_batch.fetch()
            fp.flush()
