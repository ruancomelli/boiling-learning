from __future__ import annotations

from typing import Any, Optional, Union

import tensorflow as tf

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils import PathLike, resolve
from boiling_learning.utils.dtypes import auto_spec


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[Any],
    *,
    batch_size: Optional[int] = None,
    prefetch: bool = False,
    snapshot_path: Optional[PathLike] = None,
    cache: Union[bool, PathLike] = False,
) -> tf.data.Dataset:
    sample = dataset.flatten()[0]
    typespec = auto_spec(sample)

    ds = tf.data.Dataset.from_generator(lambda: dataset, output_signature=typespec)

    if snapshot_path is not None:
        ds = ds.snapshot(str(resolve(snapshot_path, parents=True)))

    if batch_size is not None:
        ds = ds.batch(batch_size)

    if cache:
        ds = ds.cache() if isinstance(cache, bool) else ds.cache(str(resolve(cache, parents=True)))

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
