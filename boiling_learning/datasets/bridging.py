from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Union

import tensorflow as tf

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils.dtypes import auto_spec
from boiling_learning.utils.mathutils import round_to_multiple
from boiling_learning.utils.utils import PathLike, resolve


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[Any],
    *,
    batch_size: Optional[int] = None,
    filters: Iterable[Callable[[Any], bool]] = (),
    prefetch: bool = False,
    shuffle: bool = False,
    expand_to_batch_size: bool = False,
    snapshot_path: Optional[PathLike] = None,
    cache: Union[bool, PathLike] = False,
) -> tf.data.Dataset:
    if shuffle:
        dataset = dataset.shuffle()

    if prefetch:
        dataset = dataset.prefetch(batch_size)

    sample = dataset[0]
    typespec = auto_spec(sample)

    ds = tf.data.Dataset.from_generator(lambda: dataset, output_signature=typespec)

    for pred in filters:
        ds = ds.filter(pred)

    if snapshot_path is not None:
        ds = ds.snapshot(str(resolve(snapshot_path, parents=True)))

    if cache:
        # cache before batching to allow easier re-use of the same cached dataset
        ds = ds.cache() if isinstance(cache, bool) else ds.cache(str(resolve(cache, parents=True)))

    if batch_size is not None:
        if expand_to_batch_size:
            expanded_size = round_to_multiple(len(dataset), batch_size, rounder=math.ceil)
            ds = ds.repeat().take(expanded_size)

        ds = ds.batch(
            batch_size,
            drop_remainder=expand_to_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle,
        )

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
