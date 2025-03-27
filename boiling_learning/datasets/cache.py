from __future__ import annotations

import abc
import json as _json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset, SliceableDatasetCache
from boiling_learning.image_datasets import Image, Images
from boiling_learning.utils.iterutils import unsort
from boiling_learning.utils.pathutils import PathLike, resolve

_Any = TypeVar("_Any")


class NoCache(SliceableDatasetCache[_Any]):
    """A no-op cache.

    Useful to e.g. allow eager caching of upstream datasets. For instance:
    ```python
    tuple(
        dataset
        .cache(NumpyCache(...))
        .sample(Fraction(1, 100))
        # eagerly cause 1000 elements to be fetched at once, but only from the sampled
        # dataset
        .cache(EagerCache(NoCache(), 1000))
    )
    ```
    """

    def fetch_from(
        self,
        source: SliceableDataset[_Any],
        indices: Iterable[int] | None = None,
    ) -> Sequence[_Any]:
        return source.fetch(indices)

    def __repr__(self) -> str:
        return "NoCache()"


class MinimalFetchCache(SliceableDatasetCache[_Any]):
    @abc.abstractmethod
    def _store(self, pairs: dict[int, _Any]) -> None:
        pass

    @abc.abstractmethod
    def _fetch(self, indices: tuple[int, ...]) -> Sequence[_Any]:
        pass

    @abc.abstractmethod
    def _current_indices(self) -> frozenset[int]:
        pass

    def missing_indices(self, indices: Iterable[int]) -> frozenset[int]:
        return frozenset(indices) - self._current_indices()

    def fetch_from(
        self,
        source: SliceableDataset[_Any],
        indices: Iterable[int] | None = None,
    ) -> Sequence[_Any]:
        indices = tuple(range(len(source)) if indices is None else indices)

        if missing_indices := tuple(self.missing_indices(indices)):
            self._store(
                dict(
                    zip(
                        missing_indices,
                        source.fetch(missing_indices),
                    )
                )
            )

        return self._fetch(indices)


class MemoryCache(MinimalFetchCache[_Any]):
    def __init__(self) -> None:
        self._storage: dict[int, _Any] = {}

    def _store(self, pairs: dict[int, _Any]) -> None:
        self._storage.update(pairs)

    def _fetch(self, indices: tuple[int, ...]) -> tuple[_Any, ...]:
        if missing := self.missing_indices(indices).intersection(indices):
            raise ValueError(f"Required missing indices: {sorted(missing)}")

        return tuple(self._storage[index] for index in indices)

    def _current_indices(self) -> frozenset[int]:
        return frozenset(self._storage)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NumpyCache(MinimalFetchCache[Image]):
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
        logger.debug(f"Storing {len(pairs)} items {sorted(pairs)} to {self._data_path}")

        indices = list(pairs)

        data = self._data(mode="a")
        data[indices] = tuple(pairs.values())
        data.flush()

        _current_indices = self._current_indices()
        new_indices = sorted(_current_indices.union(indices))

        with self._indices_path.open("w") as file:
            _json.dump(new_indices, file)

    def _fetch(self, indices: tuple[int, ...]) -> Images:
        if missing := self.missing_indices(indices).intersection(indices):
            raise ValueError(f"Required missing indices: {sorted(missing)}")

        # sort the indices, fetch the frames and unsort them back
        unsorters, sorted_indices = unsort(indices)
        sorted_indices = list(sorted_indices)

        return self._data(mode="r")[sorted_indices][list(unsorters)]

    def _current_indices(self) -> frozenset[int]:
        if not self._indices_path.is_file():
            with self._indices_path.open("w") as file:
                _json.dump([], file)
            return frozenset()

        with self._indices_path.open("r") as file:
            return frozenset(_json.load(file))

    def _data(self, *, mode: Literal["r", "w", "a"]) -> np.memmap:
        if mode == "a" and not self._data_path.is_file():
            mode = "w"

        memmap_mode = {
            "a": "r+",
            "w": "w+",
            "r": "r",
        }[mode]

        return np.memmap(
            self._data_path,
            mode=memmap_mode,
            dtype=self._dtype,
            shape=self._shape,
        )

    @property
    def _indices_path(self) -> Path:
        return self._directory / "indices.json"

    @property
    def _data_path(self) -> Path:
        return self._directory / "data.npy"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._directory})"


class EagerCache(SliceableDatasetCache[_Any]):
    def __init__(
        self,
        wrapped: MinimalFetchCache[_Any],
        buffer_size: int,
    ) -> None:
        self._wrapped = wrapped
        self._buffer_size = buffer_size

    def fetch_from(
        self,
        source: SliceableDataset[_Any],
        indices: Iterable[int] | None = None,
    ) -> Sequence[_Any]:
        all_indices = range(len(source))
        indices = tuple(all_indices if indices is None else indices)

        if len(indices) >= self._buffer_size:
            return self._wrapped.fetch_from(source, indices)

        all_missing_indices = self._wrapped.missing_indices(all_indices)
        extra_missing_indices = all_missing_indices.difference(indices)
        eagerly_fetched_indices = (indices + tuple(extra_missing_indices))[
            : self._buffer_size
        ]
        eagerly_fetched_values = self._wrapped.fetch_from(
            source, eagerly_fetched_indices
        )
        return eagerly_fetched_values[: len(indices)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._wrapped}, buffer_size={self._buffer_size})"
