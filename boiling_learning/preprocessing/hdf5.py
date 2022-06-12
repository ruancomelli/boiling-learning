import typing
from typing import Iterable, Optional, Tuple, TypeVar

import h5py
import hdf5plugin
import more_itertools as mit
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
        with h5py.File(str(self._filepath), 'r') as file:
            frames = file[self._dataset_name][list(sorted_indices)]
            return tuple(frames[unsorter] for unsorter in unsorters)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dataset_name}@{self._filepath})'


def frames_to_hdf5(
    frames: SliceableDataset[VideoFrame],
    dest: PathLike,
    *,
    dataset_name: str,
    buffer_size: int = 1,
    open_mode: Literal['w', 'a'] = 'a',
    # experimental parameters
    indices: Optional[Iterable[int]] = None,
    compress: bool = True,
) -> None:
    """Save frames as an HDF5 file."""
    destination = str(resolve(dest))

    example_frame = frames[0]

    indices = range(len(frames)) if indices is None else tuple(indices)
    length = len(indices)

    with h5py.File(destination, open_mode) as file:
        dataset = file.require_dataset(
            dataset_name,
            (length, *example_frame.shape),
            dtype='f',
            # best compression algorithm I found - good compression, fastest decompression
            **(hdf5plugin.LZ4() if compress else {}),
        )

        for chunk_index, chunk_indices in enumerate(mit.chunked(indices, buffer_size)):
            start = chunk_index * buffer_size
            end = start + buffer_size

            logger.debug(
                f'Writing {len(chunk_indices)} frames to {destination}: ' f'{chunk_indices}'
            )

            batch = np.stack(frames.fetch(chunk_indices))
            dataset.write_direct(batch, dest_sel=np.s_[start:end])

            # flush data and close video to release memory as soon as possible
            dataset.flush()
            file.flush()
