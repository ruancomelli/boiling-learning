from functools import cache, partial
from typing import Callable, Literal, Optional, Union

import numpy as np
import tensorflow as tf
from loguru import logger

from boiling_learning.app.options import PREFETCH_BUFFER_SIZE
from boiling_learning.app.paths import analyses_path
from boiling_learning.datasets.bridging import sliceable_dataset_to_tensorflow_dataset
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.image_datasets import Image, ImageDataset, ImageDatasetTriplet, Targets
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.transforms import subset


def to_tensorflow(
    dataset: LazyDescribed[ImageDataset],
    *,
    experiment: Literal['boiling1d', 'condensation'],
    batch_size: Optional[int] = None,
    prefilterer: Optional[LazyDescribed[Callable[[Image, Targets], bool]]] = None,
    filterer: Optional[Callable[..., bool]] = None,
    target: Optional[str] = None,
    shuffle: Union[bool, int] = True,
) -> LazyDescribed[tf.data.Dataset]:
    dataset_value = dataset()

    default_prefilterer = _default_filter_for_frames_dataset(dataset_value)

    def _prefilterer(element: tuple[Image, Targets]) -> bool:
        image, targets = element
        return default_prefilterer(image, targets) and (
            prefilterer is None or prefilterer()(image, targets)
        )

    save_path = _training_datasets_allocator(experiment).allocate(dataset, prefilterer)

    logger.debug(f'Converting dataset to TF and saving to {save_path}')

    if shuffle is True:
        dataset_value = dataset_value.shuffle()

    tf_dataset = sliceable_dataset_to_tensorflow_dataset(
        dataset_value,
        # DEBUG: I commented out the following line to avoid issues with dataset saving taking too
        # long
        save_path=save_path,
        # DEBUG: try re-setting this to True
        cache=False,
        # cache=True,
        batch_size=batch_size,
        prefilterer=_prefilterer,
        filterer=filterer,
        prefetch=PREFETCH_BUFFER_SIZE,
        expand_to_batch_size=True,
        deterministic=False,
        target=target,
    )

    if shuffle and shuffle is not True:
        tf_dataset = tf_dataset.shuffle(shuffle)

    return LazyDescribed.from_value_and_description(
        tf_dataset,
        (
            dataset,
            ('prefilterer', prefilterer),
            ('batch', batch_size),
            ('target', target),
        ),
    )


def to_tensorflow_triplet(
    dataset: LazyDescribed[ImageDatasetTriplet],
    *,
    experiment: Literal['boiling1d', 'condensation'],
    batch_size: Optional[int] = None,
    prefilterer: Optional[LazyDescribed[Callable[[Image, Targets], bool]]] = None,
    filterer: Optional[Callable[..., bool]] = None,
    target: Optional[str] = None,
    include_train: bool = True,
    include_val: bool = True,
    include_test: bool = True,
    shuffle: Union[bool, int] = True,
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

    if include_train:
        logger.debug('Converting TRAIN set to tensorflow')
        ds_train = _to_tensorflow(dataset | subset('train'))
    else:
        ds_train = LazyDescribed.from_value_and_description(tf.data.Dataset.range(0), None)

    if include_val:
        logger.debug('Converting VAL set to tensorflow')
        ds_val = _to_tensorflow(dataset | subset('val'))
    else:
        ds_val = LazyDescribed.from_value_and_description(tf.data.Dataset.range(0), None)

    if include_test:
        logger.debug('Converting TEST set to tensorflow')
        ds_test = _to_tensorflow(dataset | subset('test'))
    else:
        ds_test = LazyDescribed.from_value_and_description(tf.data.Dataset.range(0), None)

    return DatasetTriplet(ds_train, ds_val, ds_test)


@cache
def _training_datasets_allocator(
    experiment: Literal['boiling1d', 'condensation']
) -> JSONAllocator:
    return JSONAllocator(analyses_path() / 'datasets' / 'training' / experiment)


def _default_filter_for_frames_dataset(
    dataset: ImageDataset,
) -> Callable[[Image, Targets], bool]:
    def _pred(image: Image, _targets: Targets) -> bool:
        first_frame, _first_targets = dataset[0]
        return image.shape == first_frame.shape and not np.allclose(image, 0)

    return _pred
