from fractions import Fraction
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.lazy import LazyDescribed, eager
from boiling_learning.preprocessing.transformers import wrap_as_partial_transformer

_Dataset = TypeVar('_Dataset', bound=SliceableDataset[Any])
_Element = TypeVar('_Element')


@wrap_as_partial_transformer
@eager
def slicer(
    datasets: DatasetTriplet[_Dataset],
    slice_: slice,
) -> DatasetTriplet[_Dataset]:
    ds_train, ds_val, ds_test = datasets
    return DatasetTriplet(ds_train[slice_], ds_val[slice_], ds_test[slice_])


@wrap_as_partial_transformer
@eager
def prefetcher(
    datasets: DatasetTriplet[_Dataset],
    buffer_size: int | None = None,
) -> DatasetTriplet[_Dataset]:
    ds_train, ds_val, ds_test = datasets
    return DatasetTriplet(
        ds_train.prefetch(buffer_size),
        ds_val.prefetch(buffer_size),
        ds_test.prefetch(buffer_size),
    )


@wrap_as_partial_transformer
@eager
def map_transformers(
    dataset: SliceableDataset[_Element],
    compiled_transformers: LazyDescribed[Callable[[_Element], _Element]],
) -> SliceableDataset[_Element]:
    return dataset.map(compiled_transformers())


@wrap_as_partial_transformer
@eager
def subset(datasets: DatasetTriplet[_Dataset], name: Literal['train', 'val', 'test']) -> _Dataset:
    ds_train, ds_val, ds_test = datasets

    return {'train': ds_train, 'val': ds_val, 'test': ds_test}[name]


@wrap_as_partial_transformer
def datasets_merger(
    datasets: tuple[LazyDescribed[DatasetTriplet[_Dataset]], ...]
) -> DatasetTriplet[_Dataset]:
    dataset_triplet = datasets_concatenater()(datasets)
    return dataset_sampler(count=Fraction(1, len(datasets)))(dataset_triplet)


@wrap_as_partial_transformer
def datasets_concatenater(
    datasets: tuple[LazyDescribed[DatasetTriplet[_Dataset]], ...]
) -> DatasetTriplet[_Dataset]:
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
    dataset_triplet: DatasetTriplet[_Dataset],
    count: Union[int, Fraction],
    subset: Optional[Literal['train', 'val', 'test']] = None,
) -> DatasetTriplet[_Dataset]:
    train, val, test = dataset_triplet

    if subset is None:
        return DatasetTriplet(train.sample(count), val.sample(count), test.sample(count))

    if subset == 'train':
        train = train.sample(count)
    elif subset == 'val':
        val = val.sample(count)
    else:
        test = test.sample(count)

    return DatasetTriplet(train, val, test)
