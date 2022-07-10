import typing
from typing import Iterable, Optional, Tuple, TypeVar

import h5py
import numpy as np
from loguru import logger
from typing_extensions import Literal

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.utils import PathLike, resolve, unsort

_T = TypeVar('_T')


class HDF5SliceableDataset(SliceableDataset[_T]):
    def __init__(self, filepath: PathLike, dataset_name: str) -> None:
        self._filepath = resolve(filepath, parents=True)
        self._dataset_name = dataset_name

    def __len__(self) -> int:
        with h5py.File(str(self._filepath), 'r') as file:
            return len(file[self._dataset_name])

    def getitem_from_index(self, index: int) -> _T:
        with h5py.File(str(self._filepath), 'r') as file:
            return typing.cast(_T, file[self._dataset_name][index])

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[_T, ...]:
        if indices is None:
            with h5py.File(str(self._filepath), 'r') as file:
                return tuple(file[self._dataset_name])

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        sorted_indices = list(sorted_indices)
        with h5py.File(str(self._filepath), 'r') as file:
            frames = file[self._dataset_name][sorted_indices]
            return tuple(frames[unsorter] for unsorter in unsorters)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dataset_name}@{self._filepath})'


def frames_to_hdf5(
    frames: SliceableDataset[VideoFrame],
    dest: PathLike,
    *,
    dataset_name: str,
    buffer_size: Optional[int] = None,
    open_mode: Literal['w', 'a'] = 'a',
) -> None:
    """Save frames as an HDF5 file."""
    destination = resolve(dest, parents=True)

    first_frame = frames[0]
    frames_count = len(frames)

    if buffer_size is None:
        buffer_size = frames_count

    with h5py.File(str(destination), open_mode) as file:
        dataset = file.require_dataset(
            dataset_name,
            (frames_count, *first_frame.shape),
            dtype=first_frame.dtype,
        )

        for chunk_index, frames_batch in enumerate(frames.batch(buffer_size)):
            n_frames = len(frames_batch)

            start = chunk_index * buffer_size
            end = start + n_frames

            logger.debug(
                f'Writing frames {start}:{end} ({n_frames}) to {dataset_name}@{destination}'
            )

            batch = np.stack(frames_batch.fetch())
            dataset.write_direct(batch, dest_sel=np.s_[start:end])

            # flush data and close video to release memory as soon as possible
            dataset.flush()
            file.flush()
