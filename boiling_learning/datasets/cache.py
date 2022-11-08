from __future__ import annotations

import abc
import json as _json
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional, Tuple, TypeVar

import numpy as np
from loguru import logger
from typing_extensions import Literal

from boiling_learning.datasets.sliceable import SliceableDataset, SliceableDatasetCache
from boiling_learning.utils.iterutils import unsort
from boiling_learning.utils.pathutils import PathLike, resolve

_Any = TypeVar('_Any')
_Array = TypeVar('_Array', bound=np.ndarray)


class _MinimalFetchCache(SliceableDatasetCache[_Any]):
    @abc.abstractmethod
    def _store(self, pairs: Dict[int, _Any]) -> None:
        pass

    @abc.abstractmethod
    def _fetch(self, indices: Tuple[int, ...]) -> Tuple[_Any, ...]:
        pass

    @abc.abstractmethod
    def _current_indices(self) -> FrozenSet[int]:
        pass

    def missing_indices(self, indices: Tuple[int, ...]) -> FrozenSet[int]:
        return frozenset(indices) - self._current_indices()

    def fetch_from(
        self,
        source: SliceableDataset[_Any],
        indices: Optional[Iterable[int]] = None,
    ) -> Tuple[_Any, ...]:
        indices = tuple(range(len(source)) if indices is None else indices)

        missing_indices = tuple(self.missing_indices(indices))
        if missing_indices:
            self._store(
                dict(
                    zip(
                        missing_indices,
                        source.fetch(missing_indices),
                    )
                )
            )

        return self._fetch(indices)


class MemoryCache(_MinimalFetchCache[_Any]):
    def __init__(self) -> None:
        self._storage: Dict[int, _Any] = {}

    def _store(self, pairs: Dict[int, _Any]) -> None:
        self._storage.update(pairs)

    def _fetch(self, indices: Tuple[int, ...]) -> Tuple[_Any, ...]:
        missing = self.missing_indices(indices).intersection(indices)
        if missing:
            raise ValueError(f'Required missing indices: {sorted(missing)}')

        return tuple(self._storage[index] for index in indices)

    def _current_indices(self) -> FrozenSet[int]:
        return frozenset(self._storage)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class NumpyCache(_MinimalFetchCache[_Array]):
    def __init__(
        self,
        directory: PathLike,
        *,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._directory = resolve(directory, dir=True)
        self._shape = shape
        self._dtype = dtype

    def _store(self, pairs: Dict[int, _Array]) -> None:
        logger.debug(f'Storing {len(pairs)} items {sorted(pairs)} to {self._data_path}')

        indices = list(pairs)

        data = self._data(mode='a')
        data[indices] = tuple(pairs.values())
        data.flush()

        _current_indices = self._current_indices()
        new_indices = sorted(_current_indices.union(indices))

        with self._indices_path.open('w') as file:
            _json.dump(new_indices, file)

    def _fetch(self, indices: Tuple[int, ...]) -> Tuple[_Array, ...]:
        missing = self.missing_indices(indices).intersection(indices)
        if missing:
            raise ValueError(f'Required missing indices: {sorted(missing)}')

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        sorted_indices = list(sorted_indices)

        frames = self._data(mode='r')[sorted_indices]
        return tuple(frames[unsorter] for unsorter in unsorters)

    def _current_indices(self) -> FrozenSet[int]:
        if not self._indices_path.is_file():
            with self._indices_path.open('w') as file:
                _json.dump([], file)
            return frozenset()

        with self._indices_path.open('r') as file:
            return frozenset(_json.load(file))

    def _data(self, *, mode: Literal['r', 'w', 'a']) -> np.memmap:
        if mode == 'a' and not self._data_path.is_file():
            mode = 'w'

        memmap_mode = {
            'a': 'r+',
            'w': 'w+',
            'r': 'r',
        }[mode]

        return np.memmap(
            self._data_path,
            mode=memmap_mode,
            dtype=self._dtype,
            shape=self._shape,
        )

    @property
    def _indices_path(self) -> Path:
        return self._directory / 'indices.json'

    @property
    def _data_path(self) -> Path:
        return self._directory / 'data.npy'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._directory})'


class EagerCache(SliceableDatasetCache[_Any]):
    def __init__(
        self,
        wrapped: _MinimalFetchCache[_Any],
        buffer_size: int,
    ) -> None:
        self._wrapped = wrapped
        self._buffer_size = buffer_size

    def fetch_from(
        self,
        source: SliceableDataset[_Any],
        indices: Optional[Iterable[int]] = None,
    ) -> Tuple[_Any, ...]:
        all_indices = range(len(source))
        indices = tuple(all_indices if indices is None else indices)

        if len(indices) >= self._buffer_size:
            return self._wrapped.fetch_from(source, indices)

        all_missing_indices = self._wrapped.missing_indices(all_indices)
        extra_missing_indices = all_missing_indices.difference(indices)
        eagerly_fetched_indices = (indices + tuple(extra_missing_indices))[: self._buffer_size]
        eagerly_fetched_values = self._wrapped.fetch_from(source, eagerly_fetched_indices)
        return eagerly_fetched_values[: len(indices)]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._wrapped}, buffer_size={self._buffer_size})'
