import tensorflow as tf

from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    DEFAULT_BOILING_OUTLIER_FILTER,
)
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed


def default_boiling_bridging(
    dataset: LazyDescribed[ImageDatasetTriplet],
) -> DatasetTriplet[LazyDescribed[tf.data.Dataset]]:
    ds_train, ds_val, ds_test = to_tensorflow_triplet(
        dataset,
        prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
        batch_size=None,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        experiment='boiling1d',
    )

    return DatasetTriplet(ds_train, ds_val, ds_test)


def default_boiling_bridging_gt10(
    dataset: LazyDescribed[ImageDatasetTriplet],
) -> DatasetTriplet[LazyDescribed[tf.data.Dataset]]:
    ds_train, ds_val, ds_test = to_tensorflow_triplet(
        dataset,
        prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
        filterer=lambda _frame, data: data[DEFAULT_BOILING_HEAT_FLUX_TARGET] >= 10,
        batch_size=None,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        experiment='boiling1d',
    )

    return DatasetTriplet(ds_train, ds_val, ds_test)
