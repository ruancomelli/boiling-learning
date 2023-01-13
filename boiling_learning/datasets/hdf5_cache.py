from __future__ import annotations

import json as _json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from boiling_learning.datasets.cache import MinimalFetchCache
from boiling_learning.image_datasets import Image, Images
from boiling_learning.utils.iterutils import unsort
from boiling_learning.utils.pathutils import PathLike, resolve


class HDF5NumpyCache(MinimalFetchCache[Image]):
    def __init__(
        self,
        directory: PathLike,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._directory = resolve(directory, dir=True)
        self._shape = shape
        self._dtype = dtype

    def _store(self, pairs: dict[int, Image]) -> None:
        logger.debug(f'Storing {len(pairs)} items {sorted(pairs)} to {self._data_path}')

        indices = list(pairs)

        with self._open_data() as data:
            data[indices] = tuple(pairs.values())

        _current_indices = self._current_indices()
        new_indices = sorted(_current_indices.union(indices))

        with self._indices_path.open('w') as file:
            _json.dump(new_indices, file)

        logger.debug('Done')

    def _fetch(self, indices: tuple[int, ...]) -> Images:
        if missing := self.missing_indices(indices).intersection(indices):
            raise ValueError(f'Required missing indices: {sorted(missing)}')

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        sorted_indices = list(sorted_indices)

        with self._open_data() as data:
            return data[sorted_indices][list(unsorters)]

    def _current_indices(self) -> frozenset[int]:
        if not self._indices_path.is_file():
            with self._indices_path.open('w') as file:
                _json.dump([], file)
            return frozenset()

        with self._indices_path.open('r') as file:
            return frozenset(_json.load(file))

    @contextmanager
    def _open_data(self, *, migrate: bool = True) -> Iterator[h5py.Dataset]:
        if migrate and self._numpy_data_path.exists():
            if not self._data_path.exists():
                logger.info(f'Migrating data from {self._numpy_data_path} to {self._data_path}')
                self._migrate_from_numpy()
                logger.info('Done')
            self._numpy_data_path.unlink()

        with h5py.File(self._data_path, 'a') as file:
            yield file.require_dataset(
                self._data_dataset_name,
                shape=self._shape,
                dtype=self._dtype,
                exact=True,
            )

    def _migrate_from_numpy(self) -> None:
        with self._open_data(migrate=False) as data:
            data[:] = np.memmap(
                self._numpy_data_path,
                mode='r',
                dtype=self._dtype,
                shape=self._shape,
            )

    @property
    def _indices_path(self) -> Path:
        return self._directory / 'indices.json'

    @property
    def _data_path(self) -> Path:
        return self._directory / 'data.h5'

    @property
    def _data_dataset_name(self) -> str:
        return 'default'

    @property
    def _numpy_data_path(self) -> Path:
        return self._directory / 'data.npy'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._directory})'
