from __future__ import annotations

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
    Dict,
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

import json_tricks as tricks
import more_itertools as mit
import numpy as np
import tensorflow as tf
from iteround import saferound
from slicerator import pipeline
from tensorflow.data import AUTOTUNE
from typing_extensions import TypeGuard

from boiling_learning.io import json
from boiling_learning.io.io import LoaderFunction, SaverFunction
from boiling_learning.io.storage import Metadata, deserialize, load, save, serialize
from boiling_learning.utils.dtypes import NestedTypeSpec, auto_spec, tf_str_dtype_bidict
from boiling_learning.utils.iterutils import distance_maximized_evenly_spaced_indices
from boiling_learning.utils.slicerators import Slicerator
from boiling_learning.utils.utils import PathLike, resolve

# pylint: disable=missing-function-docstring,missing-class-docstring

_T = TypeVar('_T')
_U = TypeVar('_U')
_X = TypeVar('_X')
_X1 = TypeVar('_X1')
_X2 = TypeVar('_X2')
_Y = TypeVar('_Y')
_Y1 = TypeVar('_Y1')
_Y2 = TypeVar('_Y2')
BooleanMask = List[bool]


class SliceableDataset(Sequence[_T]):
    def __init__(self, ancestor: Union[Sequence[_T], Slicerator[_T]] = ()) -> None:
        self._data: Slicerator[_T] = Slicerator(ancestor)

    @staticmethod
    def from_func(func: Callable[[int], _T], *, length: int) -> SliceableDataset[_T]:
        return SliceableDataset(Slicerator.from_func(func, length=length))

    @staticmethod
    def range(
        start: int, stop: Optional[int] = None, step: Optional[int] = None
    ) -> SliceableDataset[int]:
        return SliceableDataset(
            range(start, stop, step)
            if step is not None and stop is not None
            else range(start, stop)
            if stop is not None
            else range(start)
        )

    @overload
    @staticmethod
    def zip(dataset: SliceableDataset[_X], strict: bool = False) -> SliceableDataset[Tuple[_X]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[_X], __ds1: SliceableDataset[_Y], strict: bool = False
    ) -> SliceableDataset[Tuple[_X, _Y]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[_X],
        __ds1: SliceableDataset[_Y],
        __ds2: SliceableDataset[_T],
        strict: bool = False,
    ) -> SliceableDataset[Tuple[_X, _Y, _T]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[_X],
        __ds1: SliceableDataset[_Y],
        __ds2: SliceableDataset[_T],
        __ds3: SliceableDataset[_U],
        strict: bool = False,
    ) -> SliceableDataset[Tuple[_X, _Y, _T, _U]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[Any], *datasets: SliceableDataset[Any], strict: bool = False
    ) -> SliceableDataset[Tuple[Any, ...]]:
        ...

    @staticmethod
    def zip(
        dataset: SliceableDataset[Any], *datasets: SliceableDataset[Any], strict: bool = False
    ) -> SliceableDataset[Tuple[Any, ...]]:
        all_datasets = (dataset, *datasets)

        lengths = list(map(len, all_datasets))
        min_len = min(lengths)
        if strict:
            if not mit.all_equal(lengths):
                raise ValueError(f'all datasets must have the same length. Got lengths={lengths}')
        else:
            all_datasets = tuple(ds[:min_len] for ds in all_datasets)

        def getitem(i: int) -> Tuple[Any, ...]:
            return tuple(ds[i] for ds in all_datasets)

        return SliceableDataset.from_func(getitem, length=min_len)

    def __bool__(self) -> bool:
        return len(self) > 0

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
            return self.getitem_from_index(key)

        if self._is_boolean_mask(key):
            return self.getitem_from_boolean_mask(key)

        if isinstance(key, slice):
            return self.getitem_from_slice(key)

        return self.getitem_from_indices(key)

    def __iter__(self) -> Iterator[_T]:
        return iter(self._data.__iter__())

    def __len__(self) -> int:
        return len(self._data)

    def apply(
        self,
        transformation_func: Callable[[SliceableDataset[_T]], _U],
    ) -> _U:
        return transformation_func(self)

    def concatenate(self, dataset: SliceableDataset[_U]) -> SliceableDataset[Union[_T, _U]]:
        current_length = len(self)
        other_length = len(dataset)
        total_length = current_length + other_length

        def getitem(index: int) -> Union[_T, _U]:
            if index < current_length:
                return self[index]
            elif index < total_length:
                new_index = index - current_length
                return dataset[new_index]

            raise IndexError(
                f'current dataset length is {current_length}. '
                f'Other dataset length is {other_length}. '
                f'Total length is {total_length}. '
                f'Got index {index}.'
            )

        return SliceableDataset.from_func(getitem, length=total_length)

    def enumerate(self) -> SliceableDataset[Tuple[int, _T]]:
        def getitem(index: int) -> Tuple[int, _T]:
            absolute_index = _absolute_index_for_dataset(self, index)
            return absolute_index, self[absolute_index]

        return SliceableDataset.from_func(getitem, length=len(self))

    def map(
        self, __map_func: Callable[[_T], _U], *, num_parallel_calls: Optional[int] = None
    ) -> SliceableDataset[_U]:
        if num_parallel_calls is not None:
            warnings.warn(
                '`num_parallel_calls` is ignored in `SliceableDataset.map` '
                'and supported only for compatibility with `tf.data.Dataset`s'
            )

        pipeline_map = pipeline(__map_func)

        return SliceableDataset(pipeline_map(self._data))

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

        remaining: SliceableDataset[_T] = self
        splits: List[SliceableDataset[_T]] = []

        for size in int_sizes:
            splits.append(remaining.take(size))
            remaining = remaining.skip(size)

        return tuple(splits)

    def prefetch(self, _buffer_size: Optional[int] = None) -> SliceableDataset[_T]:
        warnings.warn(
            '`SliceableDataset.prefetch` is a no-op, kept only for consistency'
            ' with `tf.data.Dataset`s.'
        )
        return self

    def batch(
        self, batch_size: int, *, num_parallel_calls: Optional[int] = None
    ) -> SliceableDataset[SliceableDataset[_T]]:
        if num_parallel_calls is not None:
            warnings.warn(
                '`num_parallel_calls` is ignored in `SliceableDataset.batch` '
                'and supported only for compatibility with `tf.data.Dataset`s'
            )

        new_length: int = math.ceil(len(self) / batch_size)

        def new_data(index: int) -> SliceableDataset[_T]:
            start = index * batch_size
            end = start + batch_size
            return self[start:end]

        return SliceableDataset.from_func(new_data, length=new_length)

    def unbatch(self) -> SliceableDataset[Any]:
        return reduce(lambda left, right: left.concatenate(right), self)

    def flatten(self) -> SliceableDataset[Any]:
        flat = self
        while flat and isinstance(flat[0], SliceableDataset):
            flat = flat.unbatch()
        return flat

    @property
    def element_spec(self) -> NestedTypeSpec:
        return auto_spec(self[0])

    def getitem_from_index(self, index: int) -> _T:
        try:
            return self._data[index]
        except KeyError as e:
            raise IndexError(index, len(self)) from e

    def getitem_from_boolean_mask(self, mask: BooleanMask) -> SliceableDataset[_T]:
        if not self._is_boolean_mask(mask):
            raise ValueError(f'not a valid boolean mask: {mask}')

        indices = [index for index, boolean in enumerate(mask) if boolean]
        return self.getitem_from_indices(indices)

    def getitem_from_slice(self, slice_: slice) -> SliceableDataset[_T]:
        return self.getitem_from_indices(slice_)

    def getitem_from_indices(self, indices: Union[slice, Iterable[int]]) -> SliceableDataset[_T]:
        return SliceableDataset(self._data[indices])

    def _is_boolean_mask(self, key: Any) -> TypeGuard[BooleanMask]:
        return (
            isinstance(key, list)
            and len(key) == len(self)
            and all(isinstance(elem, bool) for elem in key)
        )


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[Any],
    *,
    batch_size: Optional[int] = None,
    prefetch: bool = False,
    shuffle: bool = False,
    snapshot_path: Optional[Path] = None,
    cache: Optional[Path] = None,
) -> tf.data.Dataset:
    sample = dataset.flatten()[0]
    typespec = auto_spec(sample)

    if shuffle:
        dataset = dataset.shuffle()

    ds = tf.data.Dataset.from_generator(lambda: dataset, output_signature=typespec)

    if snapshot_path is not None:
        ds = ds.snapshot(str(snapshot_path))

    if batch_size is not None:
        ds = ds.batch(batch_size)

    if cache is not None:
        ds = ds.cache(str(cache))

    if prefetch:
        ds = ds.prefetch(AUTOTUNE)

    return ds


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


@json.encode.instance(np.ndarray)
def _json_encode_numpy_array(obj: np.ndarray) -> str:
    return tricks.dumps(obj)


@json.decode.dispatch(np.ndarray)
def _json_decode_numpy_array(obj: str) -> np.ndarray:
    return tricks.loads(obj)


@json.encode.instance(tf.Tensor)
def _json_encode_tensor(obj: tf.Tensor) -> dict:
    return {'tensor': tf.io.serialize_tensor(obj), 'dtype': tf_str_dtype_bidict.inverse[obj.dtype]}


@json.decode.dispatch(tf.Tensor)
def _json_decode_tensor(obj: Dict[str, Any]) -> tf.Tensor:
    tensor = obj['tensor']
    dtype = tf_str_dtype_bidict[obj['dtype']]

    return tf.io.parse_tensor(tensor, dtype)


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

    return SliceableDataset.from_func(_get_element, length=spec['length'])


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
    def element_loader(path: PathLike) -> None:
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
