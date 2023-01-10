from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.image_datasets import Image, Images


class MultiMapSliceableDataset(SliceableDataset[Image]):
    def __init__(
        self,
        mapping: Callable[[Images], Images],
        ancestor: SliceableDataset[Image],
        /,
    ) -> None:
        self._ancestor = ancestor
        self._map = mapping

    def __len__(self) -> int:
        return len(self._ancestor)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._map}, {self._ancestor})'

    def getitem_from_index(self, index: int) -> Image:
        return self.fetch((index,))[0]

    def getitem_from_indices(self, indices: Iterable[int]) -> MultiMapSliceableDataset:
        return MultiMapSliceableDataset(self._map, self._ancestor[indices])

    def fetch(self, indices: Iterable[int] | None = None) -> Images:
        images = _ensure_array(self._ancestor.fetch(indices))
        return self._map(images)


def _ensure_array(sequence: Sequence[Image]) -> Images:
    return sequence if isinstance(sequence, np.ndarray) else np.array(sequence)
