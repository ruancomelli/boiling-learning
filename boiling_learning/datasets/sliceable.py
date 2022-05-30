from __future__ import annotations

import abc
import itertools
import math
import random
import warnings
from fractions import Fraction
from functools import reduce
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import more_itertools as mit
from iteround import saferound
from typing_extensions import TypeGuard, TypeVarTuple, Unpack

from boiling_learning.io import LoaderFunction, SaverFunction, json
from boiling_learning.io.storage import Metadata, deserialize, load, save, serialize
from boiling_learning.utils import PathLike, resolve
from boiling_learning.utils.dtypes import NestedTypeSpec, auto_spec
from boiling_learning.utils.iterutils import distance_maximized_evenly_spaced_indices

# pylint: disable=missing-function-docstring,missing-class-docstring

_T = TypeVar('_T')
_U = TypeVar('_U')
_X = TypeVar('_X')
_X2 = TypeVar('_X2')
_Y = TypeVar('_Y')
_Y2 = TypeVar('_Y2')
_Ts = TypeVarTuple('_Ts')
BooleanMask = List[bool]


class SliceableDataset(abc.ABC, Sequence[_T]):
    # Methods to overwrite in derived classes:
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def getitem_from_index(self, index: int) -> _T:
        pass

    # Item accessing
    @overload
    def __getitem__(self, key: int) -> _T:
        ...

    @overload
    def __getitem__(
        self, key: Union[slice, Sequence[bool], Iterable[int]]
    ) -> SliceableDataset[_T]:
        ...

    def __getitem__(
        self, key: Union[int, slice, Sequence[bool], Iterable[int]]
    ) -> Union[_T, SliceableDataset[_T]]:
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
            raise ValueError(f'not a valid boolean mask: {mask}')

        indices = range(len(self))
        filtered_indices = tuple(itertools.compress(indices, mask))
        return self.getitem_from_indices(filtered_indices)

    def getitem_from_slice(self, slice_: slice) -> SliceableDataset[_T]:
        start, stop, step = slice_.start, slice_.stop, slice_.step
        return self.getitem_from_indices(
            range(
                start if start is not None else 0,
                stop if stop is not None else len(self),
                step if step is not None else 1,
            )
        )

    # Constructors:
    @staticmethod
    def from_getitem(func: Callable[[int], _T], *, length: int) -> GetItemSliceableDataset[_T]:
        return GetItemSliceableDataset(func, length)

    @staticmethod
    def from_sequence(seq: Sequence[_T]) -> SequenceSliceableDataset[_T]:
        return SequenceSliceableDataset(seq)

    @staticmethod
    def range(
        start: int, stop: Optional[int] = None, step: Optional[int] = None
    ) -> SequenceSliceableDataset[int]:
        return SliceableDataset.from_sequence(
            range(start, stop, step)
            if step is not None and stop is not None
            else range(start, stop)
            if stop is not None
            else range(start)
        )

    @staticmethod
    def zip(
        *datasets: Unpack[Tuple[SliceableDataset[Unpack[_Ts]]]], strict: bool = True
    ) -> ZippedSliceableDataset[Unpack[_Ts]]:
        return ZippedSliceableDataset(*datasets, strict=strict)

    def concatenate(self, dataset: SliceableDataset[_U]) -> ConcatenateSliceableDataset[_T, _U]:
        return ConcatenateSliceableDataset(self, dataset)

    def map(self, __map_func: Callable[[_T], _U]) -> SliceableDataset[_U]:
        return MapSliceableDataset(__map_func, self)

    def shuffle(self) -> SliceableDataset[_T]:
        # using `random.sample` as per the docs:
        # https://docs.python.org/3/library/random.html#random.shuffle

        length = len(self)
        indices = random.sample(range(length), k=length)

        return self[indices]

    def skip(self, count: Union[int, Fraction]) -> SliceableDataset[_T]:
        if isinstance(count, int):
            return self[count:]

        total = len(self)
        keep_indices = distance_maximized_evenly_spaced_indices(
            total=total, count=total - int(count * total)
        )
        return self[keep_indices]

    def take(self, count: Union[int, Fraction]) -> SliceableDataset[_T]:
        if isinstance(count, int):
            return self[:count]

        total = len(self)
        keep_indices = distance_maximized_evenly_spaced_indices(
            total=total, count=int(count * total)
        )
        return self[keep_indices]

    @overload
    def split(
        self,
        __size1: Optional[Union[int, Fraction]],
    ) -> Tuple[SliceableDataset[_T]]:
        ...

    @overload
    def split(
        self,
        __size1: None,
        __size2: Union[int, Fraction],
    ) -> Tuple[SliceableDataset[_T], SliceableDataset[_T]]:
        ...

    @overload
    def split(
        self,
        __size1: Union[int, Fraction],
        __size2: None,
    ) -> Tuple[SliceableDataset[_T], SliceableDataset[_T]]:
        ...

    @overload
    def split(
        self,
        __size1: None,
        __size2: Union[int, Fraction],
        __size3: Union[int, Fraction],
    ) -> Tuple[SliceableDataset[_T], SliceableDataset[_T], SliceableDataset[_T]]:
        ...

    @overload
    def split(
        self,
        __size1: Union[int, Fraction],
        __size2: None,
        __size3: Union[int, Fraction],
    ) -> Tuple[SliceableDataset[_T], SliceableDataset[_T], SliceableDataset[_T]]:
        ...

    @overload
    def split(
        self,
        __size1: Union[int, Fraction],
        __size2: Union[int, Fraction],
        __size3: None,
    ) -> Tuple[SliceableDataset[_T], SliceableDataset[_T], SliceableDataset[_T]]:
        ...

    @overload
    def split(self, *sizes: Optional[Union[int, Fraction]]) -> Tuple[SliceableDataset[_T], ...]:
        ...

    def split(self, *sizes: Optional[Union[int, Fraction]]) -> Tuple[SliceableDataset[_T], ...]:
        if sizes.count(None) > 1:
            raise TypeError('`split` supports at most one `None` size.')

        length = len(self)

        rescaled_sizes: Tuple[Optional[Union[int, float]], ...] = tuple(
            float(size * length) if isinstance(size, Fraction) else size for size in sizes
        )
        total_size: Union[int, float] = sum(size for size in rescaled_sizes if size is not None)

        clean_sizes: Tuple[float, ...] = tuple(
            float(length - total_size if size is None else size) for size in rescaled_sizes
        )

        int_sizes: Tuple[int, ...] = tuple(
            map(int, saferound(clean_sizes, places=0, topline=length))
        )

        if any(size < 0 for size in int_sizes):
            raise ValueError(f'got negative sizes: {int_sizes}')

        remaining = self
        splits: List[SliceableDataset[_T]] = []

        for size in int_sizes:
            splits.append(remaining.take(size))
            remaining = remaining.skip(size)

        return tuple(splits)

    def prefetch(self, _buffer_size: Optional[int] = None) -> SliceableDataset[_T]:
        # TODO: return a custom PrefetchedDataset that allows us to better iterate over
        # it by pre-fetching `_buffer_size` items before
        warnings.warn(
            '`SliceableDataset.prefetch` is a no-op, kept only for consistency'
            ' with `tf.data.Dataset`s.'
        )
        return self

    def batch(self, batch_size: int) -> BatchSliceableDataset[_T]:
        return BatchSliceableDataset(self, batch_size)

    def unbatch(self: SliceableDataset[SliceableDataset[_U]]) -> SliceableDataset[_U]:
        return reduce(ConcatenateSliceableDataset, self)

    def flatten(self) -> SliceableDataset[Any]:
        return self.unbatch().flatten() if _is_nested_sliceable_dataset(self) else self

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[_T, ...]:
        return tuple(self[indices] if indices is not None else self)

    @property
    def element_spec(self) -> NestedTypeSpec:
        return auto_spec(self[0])

    def _is_boolean_mask(self, key: Any) -> TypeGuard[BooleanMask]:
        return (
            isinstance(key, list)
            and len(key) == len(self)
            and all(isinstance(elem, bool) for elem in key)
        )


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
        return f'{self.__class__.__name__}({self._ancestor})'


class GetItemSliceableDataset(SliceableDataset[_T]):
    def __init__(self, getitem: Callable[[int], _T], length: int) -> None:
        self._getitem = getitem
        self._length = length

    def __len__(self) -> int:
        return self._length

    def getitem_from_index(self, index: int) -> _T:
        return self._getitem(index)


class GetIndicesSliceableDataset(SliceableDataset[_T]):
    def __init__(
        self, getindices: Callable[[Iterable[int]], SliceableDataset[_T]], length: int
    ) -> None:
        self._getindices = getindices
        self._length = length

    def __len__(self) -> int:
        return self._length

    def getitem_from_index(self, index: int) -> _T:
        return mit.one(self[[index]])

    def getitem_from_indices(self, indices: Iterable[int]) -> SliceableDataset[_T]:
        return self._getindices(indices)

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[_T, ...]:
        return self._getindices(indices if indices is not None else range(len(self))).fetch()


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
        return f'{self.__class__.__name__}({self._ancestor}, {self._indices})'

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[_T, ...]:
        rebased_indices = self._indices if indices is None else self._rebase_indices(indices)
        return self._ancestor.fetch(rebased_indices)

    def _rebase_indices(self, indices: Iterable[int]) -> Tuple[int, ...]:
        return tuple(self._indices[index] for index in indices)


class ZippedSliceableDataset(SliceableDataset[Tuple[Unpack[_Ts]]], Generic[Unpack[_Ts]]):
    def __init__(self, *datasets: Unpack[Tuple[Unpack[_Ts]]], strict: bool = True) -> None:
        self._datasets = datasets

        lengths = [len(dataset) for dataset in datasets]
        if strict and not mit.all_equal(lengths):
            raise ValueError(f'all datasets must have the same length. Got lengths={lengths}')

        self._length = min(lengths)

    def __len__(self) -> int:
        return self._length

    def getitem_from_index(self, index: int) -> Tuple[Unpack[_Ts]]:
        return tuple(dataset[index] for dataset in self._datasets)

    def getitem_from_indices(self, indices: Iterable[int]) -> SliceableDataset[Tuple[Unpack[_Ts]]]:
        return ZippedSliceableDataset(*(dataset[indices] for dataset in self._datasets))

    def __repr__(self) -> str:
        reprs = ', '.join(repr(dataset) for dataset in self._datasets)
        return f'{self.__class__.__name__}({reprs})'

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[Tuple[Unpack[_Ts]]]:
        return tuple(zip(*(dataset.fetch(indices) for dataset in self._datasets)))


class ConcatenateSliceableDataset(SliceableDataset[Union[_T, _U]], Generic[_T, _U]):
    # TODO: make this variadic!
    def __init__(self, left: SliceableDataset[_T], right: SliceableDataset[_U]) -> None:
        self._left = left
        self._right = right

        self._left_length = len(self._left)
        self._right_length = len(self._right)

    def __len__(self) -> int:
        return self._left_length + self._right_length

    def getitem_from_index(self, index: int) -> Union[_T, _U]:
        if index < self._left_length:
            return self._left[index]
        else:
            return self._right[index - self._left_length]

    def getitem_from_indices(self, indices: Iterable[int]) -> ConcatenateSliceableDataset[_T, _U]:
        left_indices, right_indices = self._normalize_indices(indices)
        return ConcatenateSliceableDataset(self._left[left_indices], self._right[right_indices])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._left}, {self._right})'

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[Union[_T, _U], ...]:
        if indices is None:
            return self._left.fetch() + self._right.fetch()

        left_indices, right_indices = self._normalize_indices(indices)
        return self._left.fetch(left_indices) + self._right.fetch(right_indices)

    def _normalize_indices(self, indices: Iterable[int]) -> Tuple[List[int], List[int]]:
        left_indices = []
        right_indices = []

        for index in indices:
            if index < self._left_length:
                left_indices.append(index)
            else:
                right_indices.append(index - self._left_length)

        return left_indices, right_indices


class MapSliceableDataset(SliceableDataset[_U]):
    def __init__(self, map_func: Callable[[_T], _U], dataset: SliceableDataset[_T]) -> None:
        self._map = map_func
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def getitem_from_index(self, index: int) -> _U:
        return self._map(self._dataset[index])

    def getitem_from_indices(self, indices: Iterable[int]) -> MapSliceableDataset[_U]:
        return MapSliceableDataset(self._map, self._dataset[indices])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._map}, {self._dataset})'

    def fetch(self, indices: Optional[Iterable[int]] = None) -> Tuple[_U, ...]:
        fetched = self._dataset.fetch(indices)
        return tuple(map(self._map, fetched))


class BatchSliceableDataset(SliceableDataset[SliceableDataset[_T]], Generic[_T]):
    def __init__(self, dataset: SliceableDataset[_T], batch_size: int) -> None:
        self._dataset = dataset
        self._batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self._dataset) / self._batch_size)

    def getitem_from_index(self, index: int) -> SliceableDataset[_T]:
        start = index * self._batch_size
        end = start + self._batch_size
        return self._dataset[start:end]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dataset}, {self._batch_size})'


class SupervisedSliceableDataset(SliceableDataset[Tuple[_X, _Y]], Generic[_X, _Y]):
    @staticmethod
    def from_pairs(dataset: SliceableDataset[Tuple[_X, _Y]]) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset(dataset)

    @staticmethod
    def from_features_and_targets(
        features: SliceableDataset[_X], targets: SliceableDataset[_Y]
    ) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(SliceableDataset.zip(features, targets))

    @overload
    def __getitem__(self, key: int) -> Tuple[_X, _Y]:
        ...

    @overload
    def __getitem__(
        self, key: Union[slice, Iterable[Union[bool, int]]]
    ) -> SupervisedSliceableDataset[_X, _Y]:
        ...

    def __getitem__(
        self, key: Union[int, slice, Iterable[Union[bool, int]]]
    ) -> Union[Tuple[_X, _Y], SupervisedSliceableDataset[_X, _Y]]:
        if isinstance(key, int):
            return super().__getitem__(key)

        return SupervisedSliceableDataset.from_pairs(super().__getitem__(key))

    def map(
        self,
        __map_func: Callable[[Tuple[_X, _Y]], Tuple[_X2, _Y2]],
        *,
        num_parallel_calls: Optional[int] = None,
    ) -> SupervisedSliceableDataset[_X2, _Y2]:
        return SupervisedSliceableDataset.from_pairs(
            super().map(__map_func, num_parallel_calls=num_parallel_calls)
        )

    def shuffle(self) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().shuffle())

    def skip(self, count: Union[int, Fraction]) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().skip(count))

    def take(self, count: Union[int, Fraction]) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().take(count))

    def features(self) -> SliceableDataset[_X]:
        return super().map(itemgetter(0))

    def targets(self) -> SliceableDataset[_Y]:
        return super().map(itemgetter(1))

    def swap(self) -> SupervisedSliceableDataset[_Y, _X]:
        return self.map(itemgetter(1, 0))

    def unzip(self) -> Tuple[SliceableDataset[_X], SliceableDataset[_Y]]:
        return self.features(), self.targets()

    def map_features(self, map_func: Callable[[_X], _X2]) -> SupervisedSliceableDataset[_X2, _Y]:
        def _map_func(pair: Tuple[_X, _Y]) -> Tuple[_X2, _Y]:
            return map_func(pair[0]), pair[1]

        return self.map(_map_func)

    def map_targets(self, map_func: Callable[[_Y], _Y2]) -> SupervisedSliceableDataset[_X, _Y2]:
        def _map_func(pair: Tuple[_X, _Y]) -> Tuple[_X, _Y2]:
            return pair[0], map_func(pair[1])

        return self.map(_map_func)

    def split(
        self, *sizes: Optional[Union[int, Fraction]]
    ) -> Tuple[SupervisedSliceableDataset[_X, _Y], ...]:
        return tuple(
            SupervisedSliceableDataset.from_pairs(split) for split in super().split(*sizes)
        )


def concatenate(datasets: Iterable[SliceableDataset[_T]]) -> SliceableDataset[_T]:
    return reduce(SliceableDataset[_T].concatenate, datasets)


def save_sliceable_dataset(
    obj: SliceableDataset[_T], path: PathLike, element_saver: SaverFunction[_T] = json.dump
) -> None:
    path = resolve(path, dir=True)

    spec = {'length': len(obj)}
    json.dump(spec, path / 'spec.json')

    for index, element in enumerate(obj):
        element_saver(element, path / str(index))


def load_sliceable_dataset(
    path: PathLike, element_loader: LoaderFunction[_T] = json.load
) -> SliceableDataset[_T]:
    resolved_path: Path = resolve(path)

    spec = json.load(resolved_path / 'spec.json')
    length = spec['length']

    for index in range(length):
        if not _sliceable_dataset_element_path(resolved_path, index).exists():
            raise FileNotFoundError

    def _get_element(index: int) -> Any:
        return element_loader(_sliceable_dataset_element_path(resolved_path, index))

    return SliceableDataset.from_getitem(_get_element, length=spec['length'])


def save_supervised_sliceable_dataset(
    obj: SupervisedSliceableDataset[_X, _Y],
    path: PathLike,
    feature_saver: SaverFunction[_X],
    target_saver: SaverFunction[_Y],
) -> None:
    def element_saver(element: Tuple[_X, _Y], path: PathLike) -> None:
        resolved_path = resolve(path, dir=True)
        feature, target = element
        feature_saver(feature, resolved_path / 'feature')
        target_saver(target, resolved_path / 'target')

    save_sliceable_dataset(obj, path, element_saver)


def load_supervised_sliceable_dataset(
    path: PathLike,
    feature_loader: LoaderFunction[_X],
    target_loader: LoaderFunction[_Y],
) -> SupervisedSliceableDataset[_X, _Y]:
    def element_loader(path: PathLike) -> Tuple[_X, _Y]:
        resolved_path = resolve(path)
        feature = feature_loader(resolved_path / 'feature')
        target = target_loader(resolved_path / 'target')
        return feature, target

    return SupervisedSliceableDataset(load_sliceable_dataset(path, element_loader))


def _sliceable_dataset_element_path(root: PathLike, index: int) -> Path:
    return resolve(root) / str(index)


def _absolute_index_for_dataset(dataset: SliceableDataset[Any], index: int) -> int:
    return len(dataset) + index if index < 0 else index


@serialize.instance(SliceableDataset)
def _serialize_sliceable_dataset(instance: SliceableDataset[Any], path: Path) -> None:
    path = resolve(path, dir=True)
    save_sliceable_dataset(instance, path, element_saver=save)


@deserialize.dispatch(SliceableDataset)
def _deserialize_sliceable_dataset(
    path: Path, metadata: Metadata
) -> SliceableDataset[Any]:  # pylint: disable=unused-argument
    path = resolve(path, dir=True)
    return load_sliceable_dataset(path, element_loader=load)


@serialize.instance(SupervisedSliceableDataset)
def _serialize_supervised_sliceable_dataset(
    instance: SupervisedSliceableDataset[Any, Any], path: Path
) -> None:
    path = resolve(path, dir=True)
    save_supervised_sliceable_dataset(instance, path, feature_saver=save, target_saver=save)


@deserialize.dispatch(SupervisedSliceableDataset)
def _deserialize_supervised_sliceable_dataset(
    path: Path, metadata: Metadata
) -> SupervisedSliceableDataset[Any, Any]:  # pylint: disable=unused-argument
    path = resolve(path, dir=True)
    return load_supervised_sliceable_dataset(path, feature_loader=load, target_loader=load)


def _is_nested_sliceable_dataset(
    ds: SliceableDataset[Any],
) -> TypeGuard[SliceableDataset[SliceableDataset[Any]]]:
    return bool(ds) and isinstance(ds[0], SliceableDataset)
