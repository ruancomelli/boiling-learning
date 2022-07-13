import json as _json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, TypeVar

import numpy as np
from loguru import logger
from typing_extensions import Literal

from boiling_learning.datasets.sliceable import SliceableDatasetCache
from boiling_learning.utils.iterutils import unsort
from boiling_learning.utils.utils import PathLike, resolve

_T = TypeVar('_T')


class NumpyCache(SliceableDatasetCache[_T]):
    def __init__(self, directory: PathLike, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        self._directory = resolve(directory, dir=True)
        self._shape = shape
        self._dtype = dtype

    def store(self, pairs: Dict[int, np.ndarray]) -> None:
        logger.debug(f'Storing {len(pairs)} items {sorted(pairs)} to {self._data_path}')

        indices = list(pairs)

        data = self._data(mode='a')
        data[indices] = tuple(pairs.values())
        data.flush()

        current_indices = self._current_indices()
        new_indices = sorted(current_indices.union(indices))

        with self._indices_path.open('w') as file:
            _json.dump(new_indices, file)

    def fetch(self, indices: Optional[Iterable[int]] = None):
        if indices is None:
            indices = sorted(self._current_indices())

        indices = list(indices)

        missing = self.missing_indices(indices).intersection(indices)
        if missing:
            raise ValueError(f'Required missing indices: {sorted(missing)}')

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        sorted_indices = list(sorted_indices)

        frames = self._data(mode='r')[sorted_indices]
        return tuple(frames[unsorter] for unsorter in unsorters)

    def _data(self, *, mode: Literal['r', 'w', 'a']) -> np.memmap:
        path = self._data_path

        if mode == 'a' and not path.is_file():
            mode = 'w'

        memmap_mode = {
            'a': 'r+',
            'w': 'w+',
            'r': 'r',
        }[mode]

        return np.memmap(
            path,
            mode=memmap_mode,
            dtype=self._dtype,
            shape=self._shape,
        )

    @property
    def _data_path(self) -> Path:
        return self._directory / 'data.npy'
