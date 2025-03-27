import functools
import typing
from collections.abc import Callable
from fractions import Fraction
from typing import (
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    TypeVar,
)

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.lazy import LazyDescribed, eager
from boiling_learning.preprocessing.transformers import wrap_as_partial_transformer

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Dataset = TypeVar("_Dataset", bound=SliceableDataset[Any])
_Element = TypeVar("_Element")


def automatic_triplet_support(function: Callable[Concatenate[_Dataset, _P], _R]):
    @typing.overload
    def _wrapped(
        dataset: _Dataset,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R: ...

    @typing.overload
    def _wrapped(
        dataset: DatasetTriplet[_Dataset],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> DatasetTriplet[_R]: ...

    @functools.wraps(function)
    def _wrapped(
        dataset: _Dataset | DatasetTriplet[_Dataset],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R | DatasetTriplet[_R]:
        if isinstance(dataset, DatasetTriplet | tuple):
            ds_train, ds_val, ds_test = dataset
            return DatasetTriplet(
                function(ds_train, *args, **kwargs),
                function(ds_val, *args, **kwargs),
                function(ds_test, *args, **kwargs),
            )
        else:
            return function(dataset, *args, **kwargs)

    return _wrapped


@wrap_as_partial_transformer
@eager
@automatic_triplet_support
def slicer(
    dataset: SliceableDataset[_Element],
    slice_: slice,
) -> SliceableDataset[_Element]:
    return dataset[slice_]


@wrap_as_partial_transformer
@eager
@automatic_triplet_support
def prefetcher(
    dataset: SliceableDataset[_Element],
    buffer_size: int | None = None,
) -> SliceableDataset[_Element]:
    return dataset.prefetch(buffer_size)


@wrap_as_partial_transformer
@eager
def map_transformers(
    dataset: SliceableDataset[_Element],
    compiled_transformers: LazyDescribed[Callable[[_Element], _Element]],
) -> SliceableDataset[_Element]:
    return dataset.map(compiled_transformers())


@wrap_as_partial_transformer
@eager
def subset(
    datasets: DatasetTriplet[_Dataset], name: Literal["train", "val", "test"]
) -> _Dataset:
    ds_train, ds_val, ds_test = datasets

    return {"train": ds_train, "val": ds_val, "test": ds_test}[name]


@wrap_as_partial_transformer
def datasets_merger(
    datasets: tuple[LazyDescribed[DatasetTriplet[SliceableDataset[_Element]]], ...],
) -> DatasetTriplet[SliceableDataset[_Element]]:
    dataset_triplet = datasets_concatenater()(datasets)
    return dataset_sampler(count=Fraction(1, len(datasets)))(dataset_triplet)


@wrap_as_partial_transformer
def datasets_concatenater(
    datasets: tuple[LazyDescribed[DatasetTriplet[SliceableDataset[_Element]]], ...],
) -> DatasetTriplet[SliceableDataset[_Element]]:
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for dataset_triplet in datasets:
        ds_train, ds_val, ds_test = dataset_triplet()

        train_datasets.append(ds_train)
        val_datasets.append(ds_val)
        test_datasets.append(ds_test)

    train_dataset = SliceableDataset.concatenate(*train_datasets)
    val_dataset = SliceableDataset.concatenate(*val_datasets)
    test_dataset = SliceableDataset.concatenate(*test_datasets)

    return DatasetTriplet(train_dataset, val_dataset, test_dataset)


@wrap_as_partial_transformer
@eager
def dataset_sampler(
    dataset_triplet: DatasetTriplet[SliceableDataset[_Element]],
    count: int | Fraction,
    subset: Literal["train", "val", "test"] | None = None,
) -> DatasetTriplet[SliceableDataset[_Element]]:
    train, val, test = dataset_triplet

    if subset is None:
        return DatasetTriplet(
            train.sample(count), val.sample(count), test.sample(count)
        )

    if subset == "train":
        train = train.sample(count)
    elif subset == "val":
        val = val.sample(count)
    else:
        test = test.sample(count)

    return DatasetTriplet(train, val, test)
