from collections.abc import Callable
from functools import cache, partial
from typing import Literal

import numpy as np
import tensorflow as tf
from loguru import logger

from boiling_learning.app.constants import high_speed_cache_path
from boiling_learning.app.options import PREFETCH_BUFFER_SIZE, USE_HIGH_SPEED_CACHE
from boiling_learning.app.paths import shared_cache_path
from boiling_learning.datasets.bridging import sliceable_dataset_to_tensorflow_dataset
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.image_datasets import (
    Image,
    ImageDataset,
    ImageDatasetTriplet,
    Targets,
)
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.transforms import subset


def to_tensorflow(
    dataset: LazyDescribed[ImageDataset],
    *,
    experiment: Literal["boiling1d", "condensation"],
    batch_size: int | None = None,
    prefilterer: LazyDescribed[Callable[[Image, Targets], bool]] | None = None,
    filterer: Callable[..., bool] | None = None,
    target: str | None = None,
    shuffle: bool | int = True,
) -> LazyDescribed[tf.data.Dataset]:
    dataset_value = dataset()

    default_prefilterer = _default_filter_for_frames_dataset(dataset_value)

    def _prefilterer(element: tuple[Image, Targets]) -> bool:
        image, targets = element
        return default_prefilterer(image, targets) and (
            prefilterer is None or prefilterer()(image, targets)
        )

    if shuffle is True:
        dataset_value = dataset_value.shuffle()

    save_path = _training_datasets_allocator(experiment).allocate(
        dataset,
        prefilterer,
        shuffle,
    )

    logger.debug(f"Converting dataset to TF and saving to {save_path}")

    tf_dataset = sliceable_dataset_to_tensorflow_dataset(
        dataset_value,
        save_path=save_path,
        cache=True,
        batch_size=batch_size,
        prefilterer=_prefilterer,
        filterer=filterer,
        prefetch=PREFETCH_BUFFER_SIZE,
        deterministic=False,
        target=target,
    )

    if shuffle and shuffle is not True:
        tf_dataset = tf_dataset.shuffle(shuffle)

    return LazyDescribed.from_value_and_description(
        tf_dataset,
        (
            dataset,
            ("prefilterer", prefilterer),
            ("batch", batch_size),
            ("target", target),
        ),
    )


def to_tensorflow_triplet(
    dataset: LazyDescribed[ImageDatasetTriplet],
    *,
    experiment: Literal["boiling1d", "condensation"],
    batch_size: int | None = None,
    prefilterer: LazyDescribed[Callable[[Image, Targets], bool]] | None = None,
    filterer: Callable[..., bool] | None = None,
    target: str | None = None,
    shuffle: bool | int = True,
) -> DatasetTriplet[LazyDescribed[tf.data.Dataset]]:
    _to_tensorflow = partial(
        to_tensorflow,
        batch_size=batch_size,
        prefilterer=prefilterer,
        filterer=filterer,
        target=target,
        shuffle=shuffle,
        experiment=experiment,
    )

    logger.debug("Converting TRAIN set to tensorflow")
    ds_train = _to_tensorflow(dataset | subset("train"))
    logger.debug("Converting VAL set to tensorflow")
    ds_val = _to_tensorflow(dataset | subset("val"))
    logger.debug("Converting TEST set to tensorflow")
    ds_test = _to_tensorflow(dataset | subset("test"))
    return DatasetTriplet(ds_train, ds_val, ds_test)


@cache
def _training_datasets_allocator(
    experiment: Literal["boiling1d", "condensation"],
) -> JSONAllocator:
    cache_path = (
        high_speed_cache_path() if USE_HIGH_SPEED_CACHE else shared_cache_path()
    )
    return JSONAllocator(cache_path / "datasets" / "training" / experiment)


def _default_filter_for_frames_dataset(
    dataset: ImageDataset,
) -> Callable[[Image, Targets], bool]:
    def _pred(image: Image, _targets: Targets) -> bool:
        first_frame, _first_targets = dataset[0]
        return image.shape == first_frame.shape and not np.allclose(image, 0)

    return _pred
