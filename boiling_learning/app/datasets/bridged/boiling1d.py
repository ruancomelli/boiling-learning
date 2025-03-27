from typing import Any

import tensorflow as tf

from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet, Targets
from boiling_learning.lazy import LazyDescribed

BOILING_OUTLIER_THRESHOLD = 5
"""The threshold for classifying a data point as an outlier in the boiling dataset."""

NON_ZERO_POWER_THRESHOLD = 10
"""The threshold for classifying a data point as having non-zero power in the boiling dataset."""


def _boiling_outlier_filter(_image: Any, target: Targets) -> bool:
    return (
        abs(target["Power [W]"] - target["nominal_power"]) < BOILING_OUTLIER_THRESHOLD
    )


DEFAULT_BOILING_OUTLIER_FILTER = LazyDescribed.from_value_and_description(
    _boiling_outlier_filter,
    f"abs(Power [W] - nominal_power) < {BOILING_OUTLIER_THRESHOLD}",
)
DEFAULT_BOILING_HEAT_FLUX_TARGET = "Flux [W/cm**2]"


def default_boiling_bridging(
    dataset: LazyDescribed[ImageDatasetTriplet],
    *,
    batch_size: int | None,
    target: str = DEFAULT_BOILING_HEAT_FLUX_TARGET,
) -> DatasetTriplet[LazyDescribed[tf.data.Dataset]]:
    ds_train, ds_val, ds_test = to_tensorflow_triplet(
        dataset,
        prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
        batch_size=batch_size,
        target=target,
        experiment="boiling1d",
    )

    return DatasetTriplet(ds_train, ds_val, ds_test)


def default_boiling_bridging_gt10(
    dataset: LazyDescribed[ImageDatasetTriplet],
    *,
    batch_size: int | None,
    target: str = DEFAULT_BOILING_HEAT_FLUX_TARGET,
) -> DatasetTriplet[LazyDescribed[tf.data.Dataset]]:
    ds_train, ds_val, ds_test = to_tensorflow_triplet(
        dataset,
        prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
        filterer=(
            lambda _frame, data: data[DEFAULT_BOILING_HEAT_FLUX_TARGET]
            >= NON_ZERO_POWER_THRESHOLD
        ),
        batch_size=batch_size,
        target=target,
        experiment="boiling1d",
    )

    return DatasetTriplet(ds_train, ds_val, ds_test)
