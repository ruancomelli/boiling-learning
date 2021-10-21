from __future__ import annotations

import random
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import more_itertools as mit
import numpy as np
import tensorflow as tf
from slicerator import pipeline

from boiling_learning.utils.dtypes import auto_spec
from boiling_learning.utils.slicerators import Slicerator

_T = TypeVar('_T')
_U = TypeVar('_U')
_X = TypeVar('_X')
_Y = TypeVar('_Y')


class SliceableDataset(Sequence[_T]):
    def __init__(
        self, ancestor: Union[Sequence[_T], Slicerator[_T]] = ()
    ) -> None:
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

        return SliceableDataset(
            Slicerator.from_func(getitem, length=lenghts[0])
        )

    @overload
    def __getitem__(self, key: int) -> _T:
        ...

    @overload
    def __getitem__(
        self, key: Union[slice, Iterable[Union[bool, int]]]
    ) -> SliceableDataset[_T]:
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

    def concatenate(
        self, dataset: SliceableDataset[_U]
    ) -> SliceableDataset[Union[_T, _U]]:
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

        return SliceableDataset(
            Slicerator.from_func(new_data, length=total_length)
        )

    def enumerate(self) -> SliceableDataset[Tuple[int, _T]]:
        return SliceableDataset(Slicerator(enumerate(self), length=len(self)))

    def filter(
        self, predicate: Optional[Callable[[_T], bool]] = None
    ) -> SliceableDataset[_T]:
        return SliceableDataset(
            Slicerator(filter(predicate, self), length=len(self))
        )

    def map(self, map_func: Callable[[_T], _U]) -> SliceableDataset[_U]:
        pipeline_map = pipeline(map_func)

        return SliceableDataset(pipeline_map(self._data))

    def shuffle(self) -> SliceableDataset[_T]:
        # using `random.sample` as per the docs:
        # https://docs.python.org/3/library/random.html#random.shuffle

        length = len(self)
        indices = random.sample(range(length), k=length)

        return self[indices]

    def skip(self, count: int) -> SliceableDataset[_T]:
        return self[count:]

    def take(self, count: int) -> SliceableDataset[_T]:
        return self[:count]


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[Any],
) -> tf.data.Dataset:
    sample = dataset[0]
    typespec = auto_spec(sample)

    return tf.data.Dataset.from_generator(
        lambda: dataset, output_signature=typespec
    )


class SupervisedSliceableDataset(
    SliceableDataset[Tuple[_X, _Y]], Generic[_X, _Y]
):
    @staticmethod
    def from_pairs(
        dataset: SliceableDataset[Tuple[_X, _Y]]
    ) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset(dataset)

    @staticmethod
    def from_features_and_targets(
        features: SliceableDataset[_X], targets: SliceableDataset[_Y]
    ) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(
            SliceableDataset.zip(features, targets)
        )

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
        self, map_func: Callable[[Tuple[_X, _Y]], Tuple[_T, _U]]
    ) -> SupervisedSliceableDataset[_T, _U]:
        return SupervisedSliceableDataset.from_pairs(super().map(map_func))

    def shuffle(self) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().shuffle())

    def skip(self, count: int) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().skip(count))

    def take(self, count: int) -> SupervisedSliceableDataset[_X, _Y]:
        return SupervisedSliceableDataset.from_pairs(super().take(count))

    def features(self) -> SliceableDataset[_X]:
        return super().map(itemgetter(0))

    def targets(self) -> SliceableDataset[_Y]:
        return super().map(itemgetter(1))

    def swap(self) -> SupervisedSliceableDataset[_Y, _X]:
        return self.map(itemgetter(1, 0))

    def unzip(self) -> Tuple[SliceableDataset[_X], SliceableDataset[_Y]]:
        return self.features(), self.targets()

    def map_features(
        self, map_func: Callable[[_X], _T]
    ) -> SupervisedSliceableDataset[_T, _Y]:
        def _map_func(pair: Tuple[_X, _Y]) -> Tuple[_T, _Y]:
            return map_func(pair[0]), pair[1]

        return self.map(_map_func)

    def map_targets(
        self, map_func: Callable[[_Y], _T]
    ) -> SupervisedSliceableDataset[_X, _T]:
        def _map_func(pair: Tuple[_X, _Y]) -> Tuple[_X, _T]:
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


ImageSliceableDataset = SupervisedSliceableDataset[np.ndarray, _Y]
AnnotatedImageSliceableDataset = ImageSliceableDataset[Dict[str, Any]]
RegressionImageSliceableDataset = ImageSliceableDataset[float]
