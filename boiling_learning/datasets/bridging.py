from __future__ import annotations

import random
from typing import Any, Callable, Optional, TypeVar, Union

import tensorflow as tf

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils import PathLike, resolve
from boiling_learning.utils.dtypes import auto_spec

_T = TypeVar('_T')


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[Any],
    *,
    batch_size: Optional[int] = None,
    filter_predicate: Optional[Callable[[Any], bool]] = None,
    prefetch: bool = False,
    shuffle: bool = False,
    expand_to_batch_size: bool = False,
    snapshot_path: Optional[PathLike] = None,
    cache: Union[bool, PathLike] = False,
) -> tf.data.Dataset:
    if shuffle:
        dataset = dataset.shuffle()

    if expand_to_batch_size and batch_size is not None:
        dataset = _expand_to_batch_size(dataset, batch_size=batch_size)

    if prefetch:
        dataset = dataset.prefetch(batch_size)

    sample = dataset[0]
    typespec = auto_spec(sample)

    ds = tf.data.Dataset.from_generator(lambda: dataset, output_signature=typespec)

    if filter_predicate is not None:
        ds = ds.filter(filter_predicate)

    if snapshot_path is not None:
        ds = ds.snapshot(str(resolve(snapshot_path, parents=True)))

    if cache:
        # cache before batching to allow easier re-use of the same cached dataset
        ds = ds.cache() if isinstance(cache, bool) else ds.cache(str(resolve(cache, parents=True)))

    if batch_size is not None:
        ds = ds.batch(
            batch_size,
            drop_remainder=expand_to_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _expand_to_batch_size(
    dataset: SliceableDataset[_T], *, batch_size: int
) -> SliceableDataset[_T]:
    dataset_size = len(dataset)
    missing_count = batch_size - dataset_size % batch_size
    all_indices = range(dataset_size)
    some_indices = random.sample(all_indices, missing_count)
    return dataset.concatenate(dataset[some_indices])
