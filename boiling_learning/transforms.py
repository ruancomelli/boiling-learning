from fractions import Fraction
from typing import Any, Tuple, TypeVar, Union

from typing_extensions import Literal

from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.preprocessing.transformers import wrap_as_partial_transformer
from boiling_learning.utils.lazy import LazyDescribed, eager

_Dataset = TypeVar('_Dataset', bound=SliceableDataset[Any])


@wrap_as_partial_transformer
@eager
def subset(datasets: DatasetTriplet[_Dataset], name: Literal['train', 'val', 'test']) -> _Dataset:
    ds_train, ds_val, ds_test = datasets

    return {'train': ds_train, 'val': ds_val, 'test': ds_test}[name]


@wrap_as_partial_transformer
def datasets_merger(
    datasets: Tuple[LazyDescribed[DatasetTriplet[_Dataset]], ...]
) -> DatasetTriplet[_Dataset]:
    dataset_triplet = datasets_concatenater()(datasets)
    return dataset_sampler(count=Fraction(1, len(datasets)))(dataset_triplet)


@wrap_as_partial_transformer
def datasets_concatenater(
    datasets: Tuple[LazyDescribed[DatasetTriplet[_Dataset]], ...]
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
) -> DatasetTriplet[_Dataset]:
    train, val, test = dataset_triplet
    return DatasetTriplet(train.sample(count), val.sample(count), test.sample(count))
