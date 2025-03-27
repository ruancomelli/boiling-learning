from __future__ import annotations

import abc
import itertools
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from fractions import Fraction
from operator import itemgetter
from typing import (
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

import more_itertools as mit
from iteround import saferound
from typing_extensions import TypeVarTuple, Unpack

_T = TypeVar("_T")
_U = TypeVar("_U")
_X = TypeVar("_X")
_X2 = TypeVar("_X2")
_Y = TypeVar("_Y")
_Y2 = TypeVar("_Y2")
_Ts = TypeVarTuple("_Ts")
BooleanMask = list[bool]


class SliceableDataset(abc.ABC, Sequence[_T]):
    # Methods to overwrite in derived classes:
    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def getitem_from_index(self, index: int) -> _T:
        pass

    # Item accessing
    @overload
    def __getitem__(self, key: int) -> _T: ...

    @overload
    def __getitem__(
        self, key: slice | Sequence[bool] | Iterable[int]
    ) -> SliceableDataset[_T]: ...

    def __getitem__(
        self, key: int | slice | Sequence[bool] | Iterable[int]
    ) -> _T | SliceableDataset[_T]:
        if isinstance(key, int):
            length = len(self)
            if key >= length:
                raise IndexError(key, length)

            return self.getitem_from_index(key)

        if self._is_boolean_mask(key):
            return self.getitem_from_boolean_mask(key)

        if isinstance(key, slice):
            return self.getitem_from_slice(key)

        return self.getitem_from_indices(key)

    def getitem_from_indices(self, indices: Iterable[int]) -> SliceableDataset[_T]:
        return ComposedIndicesSliceableDataset(self, indices)

    def getitem_from_boolean_mask(self, mask: BooleanMask) -> SliceableDataset[_T]:
        if not self._is_boolean_mask(mask):
            raise ValueError(f"not a valid boolean mask: {mask}")

        indices = range(len(self))
        filtered_indices = tuple(itertools.compress(indices, mask))
        return self.getitem_from_indices(filtered_indices)

    def getitem_from_slice(self, slice_: slice) -> SliceableDataset[_T]:
        start, stop, step = slice_.start, slice_.stop, slice_.step
        return self.getitem_from_indices(
            range(
                start if start is not None else 0,
                min(stop, len(self)) if stop is not None else len(self),
                step if step is not None else 1,
            )
        )

    # Constructors:
    @staticmethod
    def concatenate(*datasets: SliceableDataset[_T]) -> ConcatenateSliceableDataset[_T]:
        return ConcatenateSliceableDataset(*datasets)

    @staticmethod
    def constantly(value: _T, *, count: int) -> SliceableDataset[_T]:
        return SliceableDataset.from_sequence((value,)).repeat(count)

    @staticmethod
    def from_getitem(
        func: Callable[[int], _T], *, length: int
    ) -> GetItemSliceableDataset[_T]:
        return GetItemSliceableDataset(func, length)

    @staticmethod
    def from_sequence(seq: Sequence[_T]) -> SequenceSliceableDataset[_T]:
        return SequenceSliceableDataset(seq)

    @staticmethod
    def range(
        start: int, stop: int | None = None, step: int | None = None
    ) -> SequenceSliceableDataset[int]:
        return SliceableDataset.from_sequence(
            range(start, stop, step)
            if step is not None and stop is not None
            else range(start, stop)
            if stop is not None
            else range(start)
        )

    @overload
    @staticmethod
    def zip(
        __dataset1: SliceableDataset[_T],
        strictness: Literal["none", "one-off", "strict"] = "strict",
    ) -> ZippedSliceableDataset[_T]: ...

    @overload
    @staticmethod
    def zip(
        __dataset1: SliceableDataset[_T],
        __dataset2: SliceableDataset[_U],
        strictness: Literal["none", "one-off", "strict"] = "strict",
    ) -> ZippedSliceableDataset[_T, _U]: ...

    @staticmethod  # type: ignore[misc] # does not handle overloads
    def zip(
        *datasets: SliceableDataset[_T],
        strictness: Literal["none", "one-off", "strict"] = "strict",
    ) -> ZippedSliceableDataset[Unpack[tuple[_T, ...]]]:
        return ZippedSliceableDataset(*datasets, strictness=strictness)

    def cache(self, cache: SliceableDatasetCache[_T]) -> CachedSliceableDataset[_T]:
        return CachedSliceableDataset(self, cache)

    def enumerate(self, *, start: int = 0) -> ZippedSliceableDataset[int, _T]:
        return SliceableDataset.zip(
            SliceableDataset.range(start, start + len(self)),
            self,
        )

    def extend(
        self, dataset: SliceableDataset[_U]
    ) -> ConcatenateSliceableDataset[_T | _U]:
        return ConcatenateSliceableDataset(self, dataset)  # type: ignore[arg-type]

    def repeat(self, count: int) -> SliceableDataset[_T]:
        return SliceableDataset.concatenate(*(self for _ in range(count)))

    def map(self, __map_func: Callable[[_T], _U]) -> SliceableDataset[_U]:
        return MapSliceableDataset(__map_func, self)

    def shuffle(self) -> SliceableDataset[_T]:
        # using `random.sample` indirectly as per the docs:
        # https://docs.python.org/3/library/random.html#random.shuffle
        return self.sample(len(self))

    def take(self, count: int | Fraction) -> SliceableDataset[_T]:
        if isinstance(count, Fraction):
            total = len(self)
            count = int(count * total)

        return self[:count]

    def skip(self, count: int | Fraction) -> SliceableDataset[_T]:
        if isinstance(count, Fraction):
            total = len(self)
            count = int(count * total)

        return self[count:]

    def sample(self, count: int | Fraction) -> SliceableDataset[_T]:
        total = len(self)

        if isinstance(count, Fraction):
            count = int(count * total)

        indices = random.sample(range(total), count)

        return self[indices]

    @overload
    def split(
        self,
        __size1: int | Fraction | None,
    ) -> tuple[SliceableDataset[_T]]: ...

    @overload
    def split(
        self,
        __size1: None,
        __size2: int | Fraction,
    ) -> tuple[SliceableDataset[_T], SliceableDataset[_T]]: ...

    @overload
    def split(
        self,
        __size1: int | Fraction,
        __size2: None,
    ) -> tuple[SliceableDataset[_T], SliceableDataset[_T]]: ...

    @overload
    def split(
        self,
        __size1: None,
        __size2: int | Fraction,
        __size3: int | Fraction,
    ) -> tuple[SliceableDataset[_T], SliceableDataset[_T], SliceableDataset[_T]]: ...

    @overload
    def split(
        self,
        __size1: int | Fraction,
        __size2: None,
        __size3: int | Fraction,
    ) -> tuple[SliceableDataset[_T], SliceableDataset[_T], SliceableDataset[_T]]: ...

    @overload
    def split(
        self,
        __size1: int | Fraction,
        __size2: int | Fraction,
        __size3: None,
    ) -> tuple[SliceableDataset[_T], SliceableDataset[_T], SliceableDataset[_T]]: ...

    @overload
    def split(
        self, *sizes: int | Fraction | None
    ) -> tuple[SliceableDataset[_T], ...]: ...

    def split(self, *sizes: int | Fraction | None) -> tuple[SliceableDataset[_T], ...]:
        if sizes.count(None) > 1:
            raise TypeError("`split` supports at most one `None` size.")

        length = len(self)

        rescaled_sizes = tuple(
            float(size * length) if isinstance(size, Fraction) else size
            for size in sizes
        )
        total_size = sum(size for size in rescaled_sizes if size is not None)

        clean_sizes = tuple(
            float(length - total_size if size is None else size)
            for size in rescaled_sizes
        )

        int_sizes = tuple(
            int(rounded_size)
            for rounded_size in saferound(clean_sizes, places=0, topline=length)
        )

        if any(size < 0 for size in int_sizes):
            raise ValueError(f"got negative sizes: {int_sizes}")

        remaining = self
        splits: list[SliceableDataset[_T]] = []

        for size in int_sizes:
            splits.append(remaining.take(size))
            remaining = remaining.skip(size)

        return tuple(splits)

    def prefetch(self, buffer_size: int | None = None) -> PrefetchedDataset[_T]:
        return PrefetchedDataset(self, buffer_size)

    def batch(self, batch_size: int) -> BatchSliceableDataset[_T]:
        return BatchSliceableDataset(self, batch_size)

    def unbatch(self: SliceableDataset[SliceableDataset[_U]]) -> SliceableDataset[_U]:
        return SliceableDataset.concatenate(*self)

    def flatten(self) -> SliceableDataset[Any]:
        return self.unbatch().flatten() if _is_nested_sliceable_dataset(self) else self

    def fetch(self, indices: Iterable[int] | None = None) -> Sequence[_T]:
        return tuple(self[indices] if indices is not None else self)

    def _is_boolean_mask(self, key: Any) -> TypeGuard[BooleanMask]:
        return (
            isinstance(key, list)
            and len(key) == len(self)
            and all(isinstance(elem, bool) for elem in key)
        )


class ProxySliceableDataset(SliceableDataset[_T]):
    def __init__(self, ancestor: SliceableDataset[_T]) -> None:
        self._ancestor = ancestor

    def __iter__(self) -> Iterator[_T]:
        return iter(self._ancestor)

    def __len__(self) -> int:
        return len(self._ancestor)

    def getitem_from_index(self, index: int) -> _T:
        return self._ancestor[index]

    def getitem_from_indices(self, indices: Iterable[int]) -> SliceableDataset[_T]:
        return self._ancestor[indices]

    def getitem_from_slice(self, slice_: slice) -> SliceableDataset[_T]:
        return self._ancestor[slice_]

    def getitem_from_boolean_mask(self, mask: BooleanMask) -> SliceableDataset[_T]:
        return self._ancestor[mask]

    def fetch(self, indices: Iterable[int] | None = None) -> Sequence[_T]:
        return self._ancestor.fetch(indices)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._ancestor})"


class SequenceSliceableDataset(SliceableDataset[_T]):
    def __init__(self, ancestor: Sequence[_T]) -> None:
        self._ancestor = ancestor

    def __iter__(self) -> Iterator[_T]:
        return iter(self._ancestor)

    def __len__(self) -> int:
        return len(self._ancestor)

    def getitem_from_index(self, index: int) -> _T:
        return self._ancestor[index]

    def getitem_from_slice(self, slice_: slice) -> SliceableDataset[_T]:
        return SequenceSliceableDataset(self._ancestor[slice_])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._ancestor})"


class GetItemSliceableDataset(SliceableDataset[_T]):
    def __init__(self, getitem: Callable[[int], _T], length: int) -> None:
        self._getitem = getitem
        self._length = length

    def __len__(self) -> int:
        return self._length

    def getitem_from_index(self, index: int) -> _T:
        return self._getitem(index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._getitem}, {self._length})"


class ComposedIndicesSliceableDataset(SliceableDataset[_T]):
    def __init__(self, ancestor: SliceableDataset[_T], indices: Iterable[int]) -> None:
        self._ancestor = ancestor
        self._indices = tuple(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def getitem_from_index(self, index: int) -> _T:
        return self._ancestor[self._indices[index]]

    def getitem_from_indices(self, indices: Iterable[int]) -> SliceableDataset[_T]:
        return self._ancestor[self._rebase_indices(indices)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._ancestor}, {self._indices})"

    def fetch(self, indices: Iterable[int] | None = None) -> Sequence[_T]:
        rebased_indices = (
            self._indices if indices is None else self._rebase_indices(indices)
        )
        return self._ancestor.fetch(rebased_indices)

    def _rebase_indices(self, indices: Iterable[int]) -> tuple[int, ...]:
        return tuple(self._indices[index] for index in indices)


class ZippedSliceableDataset(
    # due to limitations in the type system, we cannot correctly annotate the parameter
    # `datasets`, leaving `_Ts` unbound
    SliceableDataset[tuple[Unpack[_Ts]]],  # type: ignore[valid-type]
    Generic[Unpack[_Ts]],
):
    def __init__(
        self,
        *datasets: SliceableDataset[Any],
        strictness: Literal["none", "one-off", "strict"] = "strict",
    ) -> None:
        self._ancestors = datasets
        self._strictness = strictness

        lengths = [len(dataset) for dataset in datasets]
        self._check_lengths(lengths)

        self._length = min(lengths)

    def __len__(self) -> int:
        return self._length

    def getitem_from_index(self, index: int) -> tuple[Unpack[_Ts]]:
        # due to limitations in the type-system, cannot properly return the correct type
        # this is because we cannot correctly annotate `self._ancestors` to be something
        # like `(SD[_T1], SD[_T2], SD[_T3], ...)` (where `SD = SliceableDataset`).
        return tuple(dataset[index] for dataset in self._ancestors)  # type: ignore[return-value]

    def getitem_from_indices(
        self, indices: Iterable[int]
    ) -> SliceableDataset[tuple[Unpack[_Ts]]]:
        return ZippedSliceableDataset(
            *(dataset[indices] for dataset in self._ancestors)
        )

    def __repr__(self) -> str:
        reprs = ", ".join(repr(dataset) for dataset in self._ancestors)
        return f"{self.__class__.__name__}({reprs})"

    def fetch(self, indices: Iterable[int] | None = None) -> tuple[tuple[Unpack[_Ts]]]:
        if indices is not None:
            indices = tuple(indices)

        return tuple(
            zip(
                *(
                    dataset.fetch(indices)  # type: ignore[return-value]
                    for dataset in self._ancestors
                )
            )
        )

    def _check_lengths(self, lengths: list[int]) -> None:
        if self._strictness == "one-off":
            self._check_one_off(lengths)
        elif self._strictness == "strict":
            self._check_strict(lengths)

    def _check_one_off(self, lengths: list[int]) -> None:
        minimum = min(lengths)
        maximum = max(lengths)
        if maximum - minimum > 1:
            raise ValueError(
                f"[strictness={self._strictness}] "
                "Datasets may differ by at most one element. "
                f"Shorter has {minimum}, longer has {maximum}. "
                f"Got lengths={lengths}."
            )

    def _check_strict(self, lengths: list[int]) -> None:
        if not mit.all_equal(lengths):
            raise ValueError(
                f"[strictness={self._strictness}] "
                "All datasets must have the same length. "
                f"Got lengths={lengths}."
            )


class ConcatenateSliceableDataset(SliceableDataset[_T]):
    def __init__(self, *datasets: SliceableDataset[_T]) -> None:
        self._ancestors = datasets

    def __len__(self) -> int:
        return sum(len(ancestor) for ancestor in self._ancestors)

    def __repr__(self) -> str:
        ancestors = ", ".join(repr(ancestor) for ancestor in self._ancestors)
        return f"{self.__class__.__name__}({ancestors})"

    def getitem_from_index(self, index: int) -> _T:
        ancestor_index, relative_index = (
            self._absolute_index_to_ancestor_and_relative_index(index)
        )
        return self._ancestors[ancestor_index][relative_index]

    def fetch(self, indices: Iterable[int] | None = None) -> tuple[_T, ...]:
        if indices is None:
            return tuple(
                itertools.chain.from_iterable(
                    ancestor.fetch() for ancestor in self._ancestors
                )
            )

        grouped_indices: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for position, index in enumerate(indices):
            ancestor_index, relative_index = (
                self._absolute_index_to_ancestor_and_relative_index(index)
            )
            grouped_indices[ancestor_index].append((position, relative_index))

        position_value_pairs: list[tuple[int, _T]] = []
        for ancestor_index, position_relative_index_pairs in grouped_indices.items():
            if not position_relative_index_pairs:
                continue

            positions, relative_indices = mit.unzip(position_relative_index_pairs)
            values = self._ancestors[ancestor_index].fetch(relative_indices)
            position_value_pairs.extend(zip(positions, values))

        return tuple(
            value for _, value in sorted(position_value_pairs, key=itemgetter(0))
        )

    def _absolute_index_to_ancestor_and_relative_index(
        self, index: int
    ) -> tuple[int, int]:
        relative_index = index
        for ancestor_index, ancestor in enumerate(self._ancestors):
            ancestor_length = len(ancestor)
            if relative_index < ancestor_length:
                return ancestor_index, relative_index
            relative_index -= ancestor_length

        raise IndexError(index)


class MapSliceableDataset(SliceableDataset[_U]):
    def __init__(
        self, map_func: Callable[[_T], _U], dataset: SliceableDataset[_T]
    ) -> None:
        self._map = map_func
        self._ancestor = dataset

    def __iter__(self) -> Iterator[_U]:
        return (self._map(element) for element in self._ancestor)

    def __len__(self) -> int:
        return len(self._ancestor)

    def getitem_from_index(self, index: int) -> _U:
        return self._map(self._ancestor[index])

    def getitem_from_indices(self, indices: Iterable[int]) -> MapSliceableDataset[_U]:
        return MapSliceableDataset(self._map, self._ancestor[indices])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._map}, {self._ancestor})"

    def fetch(self, indices: Iterable[int] | None = None) -> tuple[_U, ...]:
        fetched = self._ancestor.fetch(indices)
        return tuple(map(self._map, fetched))


class BatchSliceableDataset(SliceableDataset[SliceableDataset[_T]], Generic[_T]):
    def __init__(self, dataset: SliceableDataset[_T], batch_size: int) -> None:
        self._ancestor = dataset
        self._batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self._ancestor) / self._batch_size)

    def getitem_from_index(self, index: int) -> SliceableDataset[_T]:
        start = index * self._batch_size
        end = start + self._batch_size
        return self._ancestor[start:end]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._ancestor}, {self._batch_size})"


class PrefetchedDataset(ProxySliceableDataset[_T]):
    def __init__(self, ancestor: SliceableDataset[_T], buffer_size: int | None) -> None:
        super().__init__(ancestor)
        self._buffer_size = buffer_size if buffer_size is not None else len(self)

    def __iter__(self) -> Iterator[_T]:
        for buffer_indices in SliceableDataset.range(len(self)).batch(
            self._buffer_size
        ):
            yield from self.fetch(buffer_indices)


class CachedSliceableDataset(SliceableDataset[_T]):
    def __init__(
        self, ancestor: SliceableDataset[_T], cache: SliceableDatasetCache[_T]
    ) -> None:
        self._ancestor = ancestor
        self._cache = cache

    def __len__(self) -> int:
        return len(self._ancestor)

    def getitem_from_index(self, index: int) -> _T:
        return self.fetch((index,))[0]

    def fetch(self, indices: Iterable[int] | None = None) -> Sequence[_T]:
        return self._cache.fetch_from(self._ancestor, indices)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._ancestor}, {self._cache})"


class SliceableDatasetCache(abc.ABC, Generic[_T]):
    @abc.abstractmethod
    def fetch_from(
        self, source: SliceableDataset[_T], indices: Iterable[int] | None = None
    ) -> Sequence[_T]:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


SupervisedSliceableDataset: TypeAlias = SliceableDataset[tuple[_X, _Y]]


def features(dataset: SupervisedSliceableDataset[_X, _Y]) -> SliceableDataset[_X]:
    # TODO: huge optimization opportunity:
    # - for ZipSliceableDataset, return the first ancestor
    # - for other types of datasets, try to inspect its parents and recurse using
    # `features`. For instance:
    #     features(ConcatenateSliceableDataset(dataset1, dataset2))
    #     == ConcatSliceableDataset(features(dataset1), features(dataset2))
    # - when impossible to recurse, use the current implementation
    return dataset.map(itemgetter(0))


def targets(dataset: SupervisedSliceableDataset[_X, _Y]) -> SliceableDataset[_Y]:
    # TODO: huge optimization opportunity: similar to `features`
    return dataset.map(itemgetter(1))


def unzip(
    dataset: SupervisedSliceableDataset[_X, _Y],
) -> tuple[SliceableDataset[_X], SliceableDataset[_Y]]:
    return features(dataset), targets(dataset)


def map_features(
    dataset: SupervisedSliceableDataset[_X, _Y], feature_mapper: Callable[[_X], _X2], /
) -> SupervisedSliceableDataset[_X2, _Y]:
    def map_func(pair: tuple[_X, _Y]) -> tuple[_X2, _Y]:
        feature, target = pair
        return feature_mapper(feature), target

    return dataset.map(map_func)


def map_targets(
    dataset: SupervisedSliceableDataset[_X, _Y], target_mapper: Callable[[_Y], _Y2], /
) -> SupervisedSliceableDataset[_X, _Y2]:
    def map_func(pair: tuple[_X, _Y]) -> tuple[_X, _Y2]:
        feature, target = pair
        return feature, target_mapper(target)

    return dataset.map(map_func)


def map_pair(
    dataset: SupervisedSliceableDataset[_X, _Y],
    feature_mapper: Callable[[_X], _X2],
    target_mapper: Callable[[_Y], _Y2],
    /,
) -> SupervisedSliceableDataset[_X2, _Y2]:
    def map_func(pair: tuple[_X, _Y]) -> tuple[_X2, _Y2]:
        feature, target = pair
        return feature_mapper(feature), target_mapper(target)

    return dataset.map(map_func)


def _is_nested_sliceable_dataset(
    ds: SliceableDataset[Any], /
) -> TypeGuard[SliceableDataset[SliceableDataset[Any]]]:
    return bool(ds) and isinstance(ds[0], SliceableDataset)
