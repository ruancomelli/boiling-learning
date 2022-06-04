import typing
from typing import Iterable, Optional, Tuple

import funcy
import h5py
import hdf5plugin
import more_itertools as mit
import numpy as np
from loguru import logger
from typing_extensions import Literal

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import Video, VideoFrame
from boiling_learning.utils.utils import PathLike, resolve, unsort


class HDF5VideoSliceableDataset(SliceableDataset[VideoFrame]):
    def __init__(self, filepath: PathLike, dataset_name: str) -> None:
        self._filepath = resolve(filepath, parents=True)
        self._dataset_name = dataset_name

    def __len__(self) -> int:
        with h5py.File(str(self._filepath), 'r', swmr=True) as file:
            return len(file[self._dataset_name])

    def getitem_from_index(self, index: int) -> VideoFrame:
        with h5py.File(str(self._filepath), 'r', swmr=True) as file:
            return typing.cast(VideoFrame, file[self._dataset_name][index])

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[VideoFrame, ...]:
        if indices is None:
            with h5py.File(str(self._filepath), 'r', swmr=True) as file:
                return tuple(file[self._dataset_name])

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        with h5py.File(str(self._filepath), 'r', swmr=True) as file:
            frames = file[self._dataset_name][list(sorted_indices)]
            return tuple(frames[unsorter] for unsorter in unsorters)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dataset_name}@{self._filepath})'


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

    length = len(video)
    chunks = (*range(0, length, batch_size), length)

    with h5py.File(destination, open_mode) as file:
        dataset = file.require_dataset(
            dataset_name,
            (length, *example_frame.shape),
            dtype='f',
            # best compression algorithm I found - good compression, fastest decompression
            **hdf5plugin.LZ4(),
        )
        for start, end in mit.pairwise(chunks):
            logger.debug(f'Writing frames {start}:{end} from {video.path} to {destination}')
            batch = np.stack([composed_transformer(video[i]) for i in range(start, end)])

            dataset.write_direct(batch, dest_sel=np.s_[start:end])

            # flush data and close video to release memory as soon as possible
            dataset.flush()
            file.flush()
            video.close()
