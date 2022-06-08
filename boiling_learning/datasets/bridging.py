from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional, TypeVar

import tensorflow as tf

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils.mathutils import round_to_multiple
from boiling_learning.utils.utils import PathLike, resolve

_T = TypeVar('_T')


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[_T],
    *,
    save_path: Optional[PathLike] = None,
    batch_size: Optional[int] = None,
    filterer: Optional[Callable[[_T], bool]] = None,
    prefetch: int = 0,
    expand_to_batch_size: bool = False,
) -> tf.data.Dataset:
    creator = partial(
        _create_tensorflow_dataset,
        dataset,
        batch_size=batch_size,
        filterer=filterer,
        prefetch=prefetch,
    )

    if save_path is None:
        ds = creator()
    else:
        save_path = str(resolve(save_path, parents=True))

        try:
            ds = tf.data.experimental.load(save_path, dataset.element_spec)
        except FileNotFoundError:
            ds = creator()
            tf.data.experimental.save(ds, save_path)
            ds = tf.data.experimental.load(save_path, dataset.element_spec)

    if batch_size is not None:
        if expand_to_batch_size:
            expanded_size = round_to_multiple(len(dataset), batch_size, rounder=math.ceil)
            ds = ds.repeat().take(expanded_size)

        ds = ds.batch(
            batch_size,
            drop_remainder=expand_to_batch_size,
            # FIXME: I am not sure if the following options were causing all sorts of
            # graph errors. Probably worth trying out:
            # num_parallel_calls=tf.data.AUTOTUNE,
            # deterministic=not shuffle,
        )

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _create_tensorflow_dataset(
    dataset: SliceableDataset[_T],
    *,
    filterer: Optional[Callable[[_T], bool]] = None,
    prefetch: int = 0,
) -> tf.data.Dataset:
    if prefetch:
        dataset = dataset.prefetch(prefetch)

    return tf.data.Dataset.from_generator(
        lambda: iter(dataset if filterer is None else filter(filterer, dataset)),
        output_signature=dataset.element_spec,
    )
