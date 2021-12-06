from __future__ import annotations

import math
import random
import warnings
from fractions import Fraction
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

from boiling_learning.io import json
from boiling_learning.utils.dtypes import NestedTypeSpec, auto_spec, tf_str_dtype_bidict
from boiling_learning.utils.iterutils import distance_maximized_evenly_spaced_indices
from boiling_learning.utils.slicerators import Slicerator
from boiling_learning.utils.utils import PathLike, resolve

_T = TypeVar('_T')
_U = TypeVar('_U')
_X = TypeVar('_X')
_X1 = TypeVar('_X1')
_X2 = TypeVar('_X2')
_Y = TypeVar('_Y')
_Y1 = TypeVar('_Y1')
_Y2 = TypeVar('_Y2')


class SliceableDataset(Sequence[_T]):
    def __init__(self, ancestor: Union[Sequence[_T], Slicerator[_T]] = ()) -> None:
        self._data: Slicerator[_T] = Slicerator(ancestor)

    @staticmethod
    def range(
        start: int, stop: Optional[int] = None, step: Optional[int] = None
    ) -> SliceableDataset[int]:
        r: range = (
            range(start, stop, step)
            if step is not None
            else range(start, stop)
            if stop is not None
            else range(start)
        )

        return SliceableDataset(r)

    @overload
    @staticmethod
    def zip(dataset: SliceableDataset[_X]) -> SliceableDataset[Tuple[_X]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[_X], __ds1: SliceableDataset[_Y]
    ) -> SliceableDataset[Tuple[_X, _Y]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[_X],
        __ds1: SliceableDataset[_Y],
        __ds2: SliceableDataset[_T],
    ) -> SliceableDataset[Tuple[_X, _Y, _T]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[_X],
        __ds1: SliceableDataset[_Y],
        __ds2: SliceableDataset[_T],
        __ds3: SliceableDataset[_U],
    ) -> SliceableDataset[Tuple[_X, _Y, _T, _U]]:
        ...

    @overload
    @staticmethod
    def zip(
        dataset: SliceableDataset[Any], *datasets: SliceableDataset[Any]
    ) -> SliceableDataset[Tuple[Any, ...]]:
        ...

    @staticmethod
    def zip(
        dataset: SliceableDataset[Any], *datasets: SliceableDataset[Any]
    ) -> SliceableDataset[Tuple[Any, ...]]:
        all_datasets = (dataset, *datasets)
        lenghts = tuple(map(len, all_datasets))

        if not mit.all_equal(lenghts):
            raise ValueError('all datasets must have the same length.')

        def getitem(i: int) -> Tuple[Any, ...]:
            return tuple(ds[i] for ds in all_datasets)

        return SliceableDataset(Slicerator.from_func(getitem, length=lenghts[0]))

    @overload
    def __getitem__(self, key: int) -> _T:
        ...

    @overload
    def __getitem__(self, key: Union[slice, Iterable[Union[bool, int]]]) -> SliceableDataset[_T]:
        ...

    def __getitem__(
        self, key: Union[int, slice, Iterable[Union[bool, int]]]
    ) -> Union[_T, SliceableDataset[_T]]:
        if isinstance(key, int):
            return self._data[key]

        return SliceableDataset(self._data[key])

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

        def new_data(index: int) -> Union[_T, _U]:
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

        return SliceableDataset(Slicerator.from_func(new_data, length=total_length))

    def enumerate(self) -> SliceableDataset[Tuple[int, _T]]:
        return SliceableDataset(Slicerator(enumerate(self), length=len(self)))

    def filter(self, predicate: Optional[Callable[[_T], bool]] = None) -> SliceableDataset[_T]:
        return SliceableDataset(Slicerator(filter(predicate, self), length=len(self)))

    def map(
        self, map_func: Callable[[_T], _U], num_parallel_calls: Optional[int] = None
    ) -> SliceableDataset[_U]:
        if num_parallel_calls is not None:
            warnings.warn(
                '`num_parallel_calls` is ignored in `SliceableDataset.map` '
                'and supported only for compatibility with `tf.data.Dataset`s'
            )

        pipeline_map = pipeline(map_func)

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

        total: int = len(self)
        keep_indices = distance_maximized_evenly_spaced_indices(
            total=total, count=total - int(count * total)
        )
        return self[keep_indices]

    def take(self, count: Union[int, Fraction]) -> SliceableDataset[_T]:
        if isinstance(count, int):
            return self[:count]

        total: int = len(self)
        keep_indices = distance_maximized_evenly_spaced_indices(
            total=total, count=int(count * total)
        )
        return self[keep_indices]

    def split(self, *sizes: Optional[Union[int, Fraction]]) -> Tuple[SliceableDataset[_T], ...]:
        if sizes.count(None) > 1:
            raise TypeError('`split` supports at most one `None` size.')

        length = len(self)

        rescaled_sizes: Tuple[Optional[Union[int, float]], ...] = tuple(
            float(size * length) if isinstance(size, Fraction) else size for size in sizes
        )
        total_size: Union[int, float] = sum(size for size in rescaled_sizes if size is not None)

        clean_sizes: Tuple[Union[int, float], ...] = tuple(
            length - total_size if size is None else size for size in rescaled_sizes
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

    def prefetch(self, buffer_size: Optional[int] = None) -> SliceableDataset[_T]:
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

        return SliceableDataset(Slicerator.from_func(new_data, length=new_length))

    @property
    def element_spec(self) -> NestedTypeSpec:
        return auto_spec(self[0])


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[Any],
) -> tf.data.Dataset:
    sample = dataset[0]
    typespec = auto_spec(sample)

    return tf.data.Dataset.from_generator(lambda: dataset, output_signature=typespec)


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

    def filter(
        self, predicate: Optional[Callable[[Tuple[_X, _Y]], bool]] = None
    ) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().filter(predicate))

    def map(
        self, map_func: Callable[[Tuple[_X, _Y]], Tuple[_X2, _Y2]]
    ) -> SupervisedSliceableDataset[_X2, _Y2]:
        return SupervisedSliceableDataset.from_pairs(super().map(map_func))

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

    def filter_features(
        self, predicate: Callable[[_X], bool]
    ) -> SupervisedSliceableDataset[_X, _Y]:
        def _filter_func(pair: Tuple[_X, _Y]) -> bool:
            return predicate(pair[0])

        return self.filter(_filter_func)

    def filter_targets(
        self, predicate: Callable[[_Y], bool]
    ) -> SupervisedSliceableDataset[_X, _Y]:
        def _filter_func(pair: Tuple[_X, _Y]) -> bool:
            return predicate(pair[1])

        return self.filter(_filter_func)

    def split(
        self, *sizes: Optional[Union[int, Fraction]]
    ) -> Tuple[SupervisedSliceableDataset[_T], ...]:
        return tuple(
            SupervisedSliceableDataset.from_pairs(split) for split in super().split(*sizes)
        )


ImageSliceableDataset = SupervisedSliceableDataset[np.ndarray, _Y]
AnnotatedImageSliceableDataset = ImageSliceableDataset[Dict[str, Any]]
RegressionImageSliceableDataset = ImageSliceableDataset[float]

PairTransformer = Callable[[_X1, _Y1], Tuple[_X2, _Y2]]
FeatureTransformer = PairTransformer[_X1, _Y, _X2, _Y]
TargetTransformer = PairTransformer[_X, _Y1, _X, _Y2]


class SupervisedSliceableDatasetPairTransformer(Generic[_X1, _Y1, _X2, _Y2]):
    def __init__(self, call: PairTransformer[_X1, _Y1, _X2, _Y2]) -> None:
        self.call: PairTransformer[_X1, _Y1, _X2, _Y2] = call

    def map_to_dataset(
        self, dataset: SupervisedSliceableDataset[_X1, _Y1]
    ) -> SupervisedSliceableDataset[_X2, _Y2]:
        return dataset.map(self.call)


class SupervisedSliceableDatasetFeatureTransformer(Generic[_X1, _X2]):
    def __init__(self, call: Callable[[_X1], _X2]) -> None:
        self.call: Callable[[_X1], _X2] = call

    def map_to_dataset(
        self, dataset: SupervisedSliceableDataset[_X1, _Y]
    ) -> SupervisedSliceableDataset[_X2, _Y]:
        return dataset.map_features(self.call)


class SupervisedSliceableDatasetTargetTransformer(Generic[_Y1, _Y2]):
    def __init__(self, call: Callable[[_Y1], _Y2]) -> None:
        self.call: Callable[[_Y1], _Y2] = call

    def map_to_dataset(
        self, dataset: SupervisedSliceableDataset[_X, _Y1]
    ) -> SupervisedSliceableDataset[_X, _Y2]:
        return dataset.map_targets(self.call)


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


def save_sliceable_dataset(obj: SliceableDataset[Any], path: PathLike) -> None:
    path = resolve(path, dir=True)

    spec = {'length': len(obj)}
    json.dump(spec, path / 'spec.json')

    for index, element in enumerate(obj):
        json.dump(element, path / f'{index}.json')


def load_sliceable_dataset(path: PathLike) -> SliceableDataset[Any]:
    resolved_path: Path = resolve(path)

    spec = json.load(resolved_path / 'spec.json')

    def _get_element(index: int) -> Any:
        return json.load(resolved_path / f'{index}.json')

    return SliceableDataset(Slicerator.from_func(_get_element, length=spec['length']))
